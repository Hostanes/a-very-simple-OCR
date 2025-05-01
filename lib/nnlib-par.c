/*
  version of nnlib.c that parallelizes each sample's operations
  parallel:
  - forward
  - backward
  - NOT softmax, mish me7erze
*/

#include "nnlib.h"
#include <CL/cl.h>

float relu(float x) { return x > 0 ? x : 0; }

float relu_derivative(float x) { return x > 0 ? 1 : 0; }

float tanh_activation(float x) { return tanhf(x); }

float tanh_derivative(float x) {
  float t = tanhf(x);
  return 1 - t * t;
}

float linear(float x) { return x; }

float linear_derivative(float x) { return 1; }

void softmax(float *array, int size) {
  // max value
  float max_val = array[0];
  for (int i = 1; i < size; i++) {
    if (array[i] > max_val) {
      max_val = array[i];
    }
  }

  // calculate sum: e^{xi - max}
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    array[i] = expf(array[i] - max_val); // Subtract max for numerical stability
    sum += array[i];
  }

  // normalize
  for (int i = 0; i < size; i++) {
    array[i] /= sum;
  }
}

void softmax_derivative(float *output, float *gradient, int size) {
  for (int i = 0; i < size; i++) {
    gradient[i] = 0;
    for (int j = 0; j < size; j++) {
      float delta = (i == j) ? 1 : 0;
      gradient[i] += output[i] * (delta - output[j]) * gradient[j];
    }
  }
}

void initialize_layer(Layer_t *layer, int input_size, int output_size,
                      ActivationFunc activation,
                      ActivationDerivative derivative) {
  layer->input_size = input_size;
  layer->output_size = output_size;
  layer->activation = activation;
  layer->activation_derivative = derivative;

  int weights_size = input_size * output_size;

  layer->weights = (float *)malloc(weights_size * sizeof(float));
  layer->biases = (float *)calloc(output_size, sizeof(float));
  layer->weight_momentum = (float *)calloc(weights_size, sizeof(float));
  layer->bias_momentum = (float *)calloc(output_size, sizeof(float));
  layer->output = (float *)calloc(output_size, sizeof(float));
  layer->input = (float *)calloc(output_size, sizeof(float));

  // Xavier init
  float scale = sqrtf(2.0f / input_size);
  randomize_weights(layer->weights, weights_size, scale);
}

void randomize_weights(float *weights, int size, float scale) {
  for (int i = 0; i < size; i++) {
    weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
  }
}

NeuralNetwork_t *create_network(int *layer_sizes, int num_layers,
                                ActivationFunc *activations,
                                ActivationDerivative *derivatives,
                                float learning_rate, float momentum) {
  NeuralNetwork_t *net = (NeuralNetwork_t *)malloc(sizeof(NeuralNetwork_t));
  net->num_layers = num_layers - 1; // First size is input layer
  net->layers = (Layer_t *)malloc(net->num_layers * sizeof(Layer_t));
  net->learning_rate = learning_rate;
  net->momentum = momentum;

  // Initialize OpenCL context
  memset(&net->cl_context, 0, sizeof(OpenCLContext_t));
  if (initialize_opencl(&net->cl_context) != 0) {
    fprintf(stderr, "ERROR: OpenCL init failed\n");
  }

  for (int i = 0; i < net->num_layers; i++) {
    initialize_layer(&net->layers[i], layer_sizes[i], layer_sizes[i + 1],
                     activations[i], derivatives[i]);
  }

  return net;
}

void free_network(NeuralNetwork_t *net) {
  for (int i = 0; i < net->num_layers; i++) {
    free(net->layers[i].weights);
    free(net->layers[i].biases);
    free(net->layers[i].weight_momentum);
    free(net->layers[i].bias_momentum);
    free(net->layers[i].output);
    free(net->layers[i].input);
  }
  free(net->layers);

  // Cleanup OpenCL
  if (net->cl_context.opencl_initialized) {
    cleanup_opencl(&net->cl_context);
  }

  free(net);
}

char *load_kernel_source(const char *filename, size_t *source_size) {
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    perror("Failed to open kernel file");
    return NULL;
  }

  // Get file size
  fseek(fp, 0, SEEK_END);
  *source_size = ftell(fp);
  rewind(fp);

  // Allocate buffer
  char *source = (char *)malloc(*source_size + 1);
  if (!source) {
    perror("Failed to allocate memory for kernel source");
    fclose(fp);
    return NULL;
  }

  // Read file
  size_t read_size = fread(source, 1, *source_size, fp);
  if (read_size != *source_size) {
    perror("Failed to read kernel source");
    free(source);
    fclose(fp);
    return NULL;
  }

  source[*source_size] = '\0'; // Null-terminate
  fclose(fp);
  return source;
}

int initialize_opencl(OpenCLContext_t *context) {
  cl_int err;

  // Get platform
  err = clGetPlatformIDs(1, &context->platform, NULL);
  if (err != CL_SUCCESS)
    return -1;

  // Get GPU device
  err = clGetDeviceIDs(context->platform, CL_DEVICE_TYPE_GPU, 1,
                       &context->device, NULL);
  if (err != CL_SUCCESS)
    return -1;

  // Create context
  context->context =
      clCreateContext(NULL, 1, &context->device, NULL, NULL, &err);
  if (err != CL_SUCCESS)
    return -1;

  // Create command queue
  context->queue = clCreateCommandQueueWithProperties(context->context,
                                                      context->device, 0, &err);
  if (err != CL_SUCCESS)
    return -1;

  // Load kernel source from file
  size_t source_size;
  char *kernel_source = load_kernel_source("nn-kernel.cl", &source_size);
  if (!kernel_source) {
    clReleaseCommandQueue(context->queue);
    clReleaseContext(context->context);
    return -1;
  }

  context->program = clCreateProgramWithSource(
      context->context, 1, (const char **)&kernel_source, NULL, &err);

  free(kernel_source);

  if (err != CL_SUCCESS)
    return -1;

  // Build program
  err = clBuildProgram(context->program, 1, &context->device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    // Get build log
    size_t log_size;
    clGetProgramBuildInfo(context->program, context->device,
                          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = (char *)malloc(log_size);
    clGetProgramBuildInfo(context->program, context->device,
                          CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    fprintf(stderr, "Build error:\n%s\n", log);
    free(log);

    clReleaseProgram(context->program);
    clReleaseCommandQueue(context->queue);
    clReleaseContext(context->context);
    return -1;
  }

  // Create all kernels
  context->matrix_dot_kernel =
      clCreateKernel(context->program, "matrix_dot_product", &err);
  context->activation_kernel =
      clCreateKernel(context->program, "apply_activation", &err);
  context->output_gradient_kernel =
      clCreateKernel(context->program, "output_layer_gradient", &err);
  context->hidden_gradient_kernel =
      clCreateKernel(context->program, "hidden_layer_gradient", &err);
  context->update_weights_kernel =
      clCreateKernel(context->program, "update_weights", &err);
  context->update_biases_kernel =
      clCreateKernel(context->program, "update_biases", &err);

  if (err != CL_SUCCESS) {
    fprintf(stderr, "ERROR: creating the kernels %d\n", err);
    clReleaseKernel(context->matrix_dot_kernel);
    clReleaseKernel(context->activation_kernel);
    clReleaseKernel(context->output_gradient_kernel);
    clReleaseKernel(context->hidden_gradient_kernel);
    clReleaseKernel(context->update_weights_kernel);
    clReleaseKernel(context->update_biases_kernel);
    clReleaseProgram(context->program);
    clReleaseCommandQueue(context->queue);
    clReleaseContext(context->context);
    return -1;
  }

  context->opencl_initialized = 1;
  return 0;

  context->opencl_initialized = 1;
  return 0;
}

void cleanup_opencl(OpenCLContext_t *context) {
  if (context->matrix_dot_kernel)
    clReleaseKernel(context->matrix_dot_kernel);
  if (context->program)
    clReleaseProgram(context->program);
  if (context->queue)
    clReleaseCommandQueue(context->queue);
  if (context->context)
    clReleaseContext(context->context);
  context->opencl_initialized = 0;
}

float *forward_pass(NeuralNetwork_t *net, float *input) {
  float *current_input = input;

  for (int i = 0; i < net->num_layers; i++) {
    Layer_t *layer = &net->layers[i];

    // Use OpenCL for matrix multiplication if available
    if (net->cl_context.opencl_initialized) {
      cl_int err;
      size_t global_size = layer->output_size;

      // Create buffers
      cl_mem input_buf = clCreateBuffer(
          net->cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          layer->input_size * sizeof(float), current_input, &err);
      if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating input buffer: %d\n", err);
        return NULL;
      }

      cl_mem weights_buf = clCreateBuffer(
          net->cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          layer->input_size * layer->output_size * sizeof(float),
          layer->weights, &err);
      if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating weights buffer: %d\n", err);
        clReleaseMemObject(input_buf);
        return NULL;
      }

      cl_mem output_buf =
          clCreateBuffer(net->cl_context.context, CL_MEM_READ_WRITE,
                         layer->output_size * sizeof(float), NULL, &err);
      if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating output buffer: %d\n", err);
        clReleaseMemObject(input_buf);
        clReleaseMemObject(weights_buf);
        return NULL;
      }

      cl_mem biases_buf = clCreateBuffer(
          net->cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          layer->output_size * sizeof(float), layer->biases, &err);
      if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating biases buffer: %d\n", err);
        clReleaseMemObject(input_buf);
        clReleaseMemObject(weights_buf);
        clReleaseMemObject(output_buf);
        return NULL;
      }

      // Set matrix multiplication kernel arguments
      err = clSetKernelArg(net->cl_context.matrix_dot_kernel, 0, sizeof(cl_mem),
                           &input_buf);
      err |= clSetKernelArg(net->cl_context.matrix_dot_kernel, 1,
                            sizeof(cl_mem), &weights_buf);
      err |= clSetKernelArg(net->cl_context.matrix_dot_kernel, 2,
                            sizeof(cl_mem), &output_buf);
      err |= clSetKernelArg(net->cl_context.matrix_dot_kernel, 3,
                            sizeof(cl_mem), &biases_buf);
      err |= clSetKernelArg(net->cl_context.matrix_dot_kernel, 4, sizeof(int),
                            &layer->input_size);
      err |= clSetKernelArg(net->cl_context.matrix_dot_kernel, 5, sizeof(int),
                            &layer->output_size);

      if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting kernel arguments: %d\n", err);
        clReleaseMemObject(input_buf);
        clReleaseMemObject(weights_buf);
        clReleaseMemObject(output_buf);
        clReleaseMemObject(biases_buf);
        return NULL;
      }

      // Run matrix multiplication kernel
      err = clEnqueueNDRangeKernel(net->cl_context.queue,
                                   net->cl_context.matrix_dot_kernel, 1, NULL,
                                   &global_size, NULL, 0, NULL, NULL);
      if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueueing matrix kernel: %d\n", err);
        clReleaseMemObject(input_buf);
        clReleaseMemObject(weights_buf);
        clReleaseMemObject(output_buf);
        clReleaseMemObject(biases_buf);
        return NULL;
      }

      // Read back matrix multiplication results to layer->input
      err = clEnqueueReadBuffer(net->cl_context.queue, output_buf, CL_TRUE, 0,
                                layer->output_size * sizeof(float),
                                layer->input, 0, NULL, NULL);
      if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading output buffer: %d\n", err);
        clReleaseMemObject(input_buf);
        clReleaseMemObject(weights_buf);
        clReleaseMemObject(output_buf);
        clReleaseMemObject(biases_buf);
        return NULL;
      }

      // Release buffers we don't need anymore
      clReleaseMemObject(input_buf);
      clReleaseMemObject(weights_buf);
      clReleaseMemObject(biases_buf);

      // Handle activation function
      if (layer->activation == softmax_placeholder) {
        // For softmax, we need to do CPU computation
        memcpy(layer->output, layer->input, layer->output_size * sizeof(float));
        softmax(layer->output, layer->output_size);
      } else {
        // For other activations, use OpenCL
        int activation_type;
        if (layer->activation == relu) {
          activation_type = 0; // RELU
        } else if (layer->activation == sigmoid) {
          activation_type = 1; // SIGMOID
        } else if (layer->activation == tanh_activation) {
          activation_type = 2; // TANH
        } else {
          activation_type = 3; // LINEAR
        }

        // Create output buffer for activation
        cl_mem output_act_buf = clCreateBuffer(
            net->cl_context.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            layer->output_size * sizeof(float), layer->output, &err);
        if (err != CL_SUCCESS) {
          fprintf(stderr, "Error creating activation output buffer: %d\n", err);
          clReleaseMemObject(output_buf);
          return NULL;
        }

        // Set activation kernel arguments
        err = clSetKernelArg(net->cl_context.activation_kernel, 0,
                             sizeof(cl_mem), &output_buf);
        err |= clSetKernelArg(net->cl_context.activation_kernel, 1,
                              sizeof(cl_mem), &output_act_buf);
        err |= clSetKernelArg(net->cl_context.activation_kernel, 2, sizeof(int),
                              &layer->output_size);
        err |= clSetKernelArg(net->cl_context.activation_kernel, 3, sizeof(int),
                              &activation_type);

        if (err != CL_SUCCESS) {
          fprintf(stderr, "Error setting activation kernel args: %d\n", err);
          clReleaseMemObject(output_buf);
          clReleaseMemObject(output_act_buf);
          return NULL;
        }

        // Run activation kernel
        err = clEnqueueNDRangeKernel(net->cl_context.queue,
                                     net->cl_context.activation_kernel, 1, NULL,
                                     &global_size, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
          fprintf(stderr, "Error enqueueing activation kernel: %d\n", err);
          clReleaseMemObject(output_buf);
          clReleaseMemObject(output_act_buf);
          return NULL;
        }

        // Read back activation results
        err = clEnqueueReadBuffer(
            net->cl_context.queue, output_act_buf, CL_TRUE, 0,
            layer->output_size * sizeof(float), layer->output, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
          fprintf(stderr, "Error reading activation output: %d\n", err);
        }

        // Release buffers
        clReleaseMemObject(output_buf);
        clReleaseMemObject(output_act_buf);
      }
    } else {
      // CPU fallback implementation
      for (int j = 0; j < layer->output_size; j++) {
        float sum = layer->biases[j];
        for (int k = 0; k < layer->input_size; k++) {
          sum += current_input[k] * layer->weights[k * layer->output_size + j];
        }
        layer->input[j] = sum;
      }

      // Activation
      if (layer->activation == softmax_placeholder) {
        memcpy(layer->output, layer->input, layer->output_size * sizeof(float));
        softmax(layer->output, layer->output_size);
      } else {
        for (int j = 0; j < layer->output_size; j++) {
          layer->output[j] = layer->activation(layer->input[j]);
        }
      }
    }

    current_input = layer->output;
  }
  return current_input;
}

void backward_pass(NeuralNetwork_t *net, float *input, float *target) {
  float *output = forward_pass(net, input);
  Layer_t *output_layer = &net->layers[net->num_layers - 1];

  // Allocate OpenCL buffers for deltas
  cl_mem *delta_bufs = malloc(net->num_layers * sizeof(cl_mem));
  for (int i = 0; i < net->num_layers; i++) {
    delta_bufs[i] =
        clCreateBuffer(net->cl_context.context, CL_MEM_READ_WRITE,
                       net->layers[i].output_size * sizeof(float), NULL, &err);
  }

  if (net->cl_context.opencl_initialized) {
    // Output layer gradient
    int activation_type =
        (output_layer->activation == softmax_placeholder) ? 4 : 0;
    clSetKernelArg(net->cl_context.output_gradient_kernel, 0, sizeof(cl_mem),
                   &output_layer->output_buf);
    clSetKernelArg(net->cl_context.output_gradient_kernel, 1, sizeof(cl_mem),
                   &target_buf);
    clSetKernelArg(net->cl_context.output_gradient_kernel, 2, sizeof(cl_mem),
                   &delta_bufs[net->num_layers - 1]);
    clSetKernelArg(net->cl_context.output_gradient_kernel, 3, sizeof(int),
                   &output_layer->output_size);
    clSetKernelArg(net->cl_context.output_gradient_kernel, 4, sizeof(int),
                   &activation_type);

    size_t global_size = output_layer->output_size;
    clEnqueueNDRangeKernel(net->cl_context.queue,
                           net->cl_context.output_gradient_kernel, 1, NULL,
                           &global_size, NULL, 0, NULL, NULL);

    // Hidden layers gradients
    for (int l = net->num_layers - 2; l >= 0; l--) {
      Layer_t *current = &net->layers[l];
      Layer_t *next = &net->layers[l + 1];

      clSetKernelArg(net->cl_context.hidden_gradient_kernel, 0, sizeof(cl_mem),
                     &next->weights_buf);
      clSetKernelArg(net->cl_context.hidden_gradient_kernel, 1, sizeof(cl_mem),
                     &delta_bufs[l + 1]);
      clSetKernelArg(net->cl_context.hidden_gradient_kernel, 2, sizeof(cl_mem),
                     &current->output_buf);
      clSetKernelArg(net->cl_context.hidden_gradient_kernel, 3, sizeof(cl_mem),
                     &delta_bufs[l]);
      clSetKernelArg(net->cl_context.hidden_gradient_kernel, 4, sizeof(int),
                     &current->output_size);
      clSetKernelArg(net->cl_context.hidden_gradient_kernel, 5, sizeof(int),
                     &next->output_size);
      clSetKernelArg(net->cl_context.hidden_gradient_kernel, 6, sizeof(int),
                     &activation_type);

      global_size = current->output_size;
      clEnqueueNDRangeKernel(net->cl_context.queue,
                             net->cl_context.hidden_gradient_kernel, 1, NULL,
                             &global_size, NULL, 0, NULL, NULL);
    }

    // Update weights and biases
    float *prev_output = input;
    for (int l = 0; l < net->num_layers; l++) {
      Layer_t *layer = &net->layers[l];

      // Update weights
      size_t global_work_size[2] = {layer->input_size, layer->output_size};
      clSetKernelArg(net->cl_context.update_weights_kernel, 0, sizeof(cl_mem),
                     &layer->weights_buf);
      clSetKernelArg(net->cl_context.update_weights_kernel, 1, sizeof(cl_mem),
                     &layer->weight_momentum_buf);
      clSetKernelArg(net->cl_context.update_weights_kernel, 2, sizeof(cl_mem),
                     &prev_output_buf);
      clSetKernelArg(net->cl_context.update_weights_kernel, 3, sizeof(cl_mem),
                     &delta_bufs[l]);
      clSetKernelArg(net->cl_context.update_weights_kernel, 4, sizeof(float),
                     &net->learning_rate);
      clSetKernelArg(net->cl_context.update_weights_kernel, 5, sizeof(float),
                     &net->momentum);
      clSetKernelArg(net->cl_context.update_weights_kernel, 6, sizeof(int),
                     &layer->input_size);
      clSetKernelArg(net->cl_context.update_weights_kernel, 7, sizeof(int),
                     &layer->output_size);

      clEnqueueNDRangeKernel(net->cl_context.queue,
                             net->cl_context.update_weights_kernel, 2, NULL,
                             global_work_size, NULL, 0, NULL, NULL);

      // Update biases
      global_size = layer->output_size;
      clSetKernelArg(net->cl_context.update_biases_kernel, 0, sizeof(cl_mem),
                     &layer->biases_buf);
      clSetKernelArg(net->cl_context.update_biases_kernel, 1, sizeof(cl_mem),
                     &layer->bias_momentum_buf);
      clSetKernelArg(net->cl_context.update_biases_kernel, 2, sizeof(cl_mem),
                     &delta_bufs[l]);
      clSetKernelArg(net->cl_context.update_biases_kernel, 3, sizeof(float),
                     &net->learning_rate);
      clSetKernelArg(net->cl_context.update_biases_kernel, 4, sizeof(float),
                     &net->momentum);
      clSetKernelArg(net->cl_context.update_biases_kernel, 5, sizeof(int),
                     &layer->output_size);

      clEnqueueNDRangeKernel(net->cl_context.queue,
                             net->cl_context.update_biases_kernel, 1, NULL,
                             &global_size, NULL, 0, NULL, NULL);

      prev_output = layer->output;
    }
  } else {
    // CPU fallback
    // ... existing CPU code ...
  }

  // Cleanup
  for (int i = 0; i < net->num_layers; i++) {
    clReleaseMemObject(delta_bufs[i]);
  }
  free(delta_bufs);
}

void train(NeuralNetwork_t *net, float *input, float *target) {
  backward_pass(net, input, target); // backward pass includes forward pass
                                     // function call, TODO fix this
}

int predict(NeuralNetwork_t *net, float *input) {
  float *output = forward_pass(net, input);
  Layer_t *output_layer = &net->layers[net->num_layers - 1];

  int max_index = 0;
  for (int i = 1; i < output_layer->output_size; i++) {
    if (output[i] > output[max_index]) {
      max_index = i;
    }
  }

  return max_index;
}

// ===================================
//    for parallelized training loop
// ===================================

// Computes gradients without applying updates
void compute_gradients(NeuralNetwork_t *net, float *input, float *target,
                       float **gradients, float **bias_gradients) {
  // Forward pass (same as original)
  float *output = forward_pass(net, input);
  Layer_t *output_layer = &net->layers[net->num_layers - 1];

  // Allocate deltas
  float **deltas = malloc(net->num_layers * sizeof(float *));
  for (int i = 0; i < net->num_layers; i++) {
    deltas[i] = calloc(net->layers[i].output_size, sizeof(float));
  }

  // Output layer gradient (same as original)
  if (output_layer->activation == softmax_placeholder) {
    for (int i = 0; i < output_layer->output_size; i++) {
      deltas[net->num_layers - 1][i] = output_layer->output[i] - target[i];
    }
  } else {
    for (int i = 0; i < output_layer->output_size; i++) {
      float error = output_layer->output[i] - target[i];
      deltas[net->num_layers - 1][i] =
          error * output_layer->activation_derivative(output_layer->input[i]);
    }
  }

  // Backpropagation (same as original)
  for (int l = net->num_layers - 2; l >= 0; l--) {
    Layer_t *current = &net->layers[l];
    Layer_t *next = &net->layers[l + 1];

    for (int i = 0; i < current->output_size; i++) {
      float error = 0;
      for (int j = 0; j < next->output_size; j++) {
        error += next->weights[i * next->output_size + j] * deltas[l + 1][j];
      }
      deltas[l][i] = error * current->activation_derivative(current->input[i]);
    }
  }

  // Accumulate gradients instead of applying updates
  float *prev_output = input;
  for (int l = 0; l < net->num_layers; l++) {
    Layer_t *layer = &net->layers[l];

    // Accumulate weight gradients
    for (int i = 0; i < layer->input_size; i++) {
      for (int j = 0; j < layer->output_size; j++) {
        int idx = i * layer->output_size + j;
        gradients[l][idx] += prev_output[i] * deltas[l][j];
      }
    }

    // Accumulate bias gradients
    for (int j = 0; j < layer->output_size; j++) {
      bias_gradients[l][j] += deltas[l][j];
    }

    prev_output = layer->output;
  }

  // Clean up
  for (int i = 0; i < net->num_layers; i++) {
    free(deltas[i]);
  }
  free(deltas);
}

// Applies accumulated gradients
void apply_updates(NeuralNetwork_t *net, float **gradients,
                   float **bias_gradients, int batch_size) {
  for (int l = 0; l < net->num_layers; l++) {
    Layer_t *layer = &net->layers[l];
    int weights_size = layer->input_size * layer->output_size;

    // Update weights with momentum
    for (int i = 0; i < weights_size; i++) {
      float avg_gradient = gradients[l][i] / batch_size;
      layer->weight_momentum[i] = net->momentum * layer->weight_momentum[i] +
                                  net->learning_rate * avg_gradient;
      layer->weights[i] -= layer->weight_momentum[i];
    }

    // Update biases with momentum
    for (int j = 0; j < layer->output_size; j++) {
      float avg_gradient = bias_gradients[l][j] / batch_size;
      layer->bias_momentum[j] = net->momentum * layer->bias_momentum[j] +
                                net->learning_rate * avg_gradient;
      layer->biases[j] -= layer->bias_momentum[j];
    }
  }
}

/*
  Caclulates loss of a layer depending on its activation function
  Cross-entropy for softmax
  MSE for ReLU, or anything else
*/
float calculate_loss(NeuralNetwork_t *net, float *output, float *target) {
  Layer_t *output_layer = &net->layers[net->num_layers - 1];
  float loss = 0;

  // cross-entropy loss
  if (output_layer->activation == softmax_placeholder) {
    for (int i = 0; i < output_layer->output_size; i++) {
      if (target[i] > 0) {
        loss += -target[i] * logf(output[i] + 1e-10f);
      }
    }
  } else {
    // MSE loss
    for (int i = 0; i < output_layer->output_size; i++) {
      float diff = output[i] - target[i];
      loss += diff * diff;
    }
    loss /= output_layer->output_size;
  }

  return loss;
}

float softmax_placeholder(float x) { return x; }

int save_Network(NeuralNetwork_t *network, const char *filename) {
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    perror("Failed to open file for writing");
    return -1;
  }

  // Header
  uint8_t magic = MAGIC_NUMBER;
  uint32_t version = FILE_VERSION;
  fwrite(&magic, sizeof(uint8_t), 1, fp);
  fwrite(&version, sizeof(uint32_t), 1, fp);

  // Network configuration
  fwrite(&network->learning_rate, sizeof(float), 1, fp);
  fwrite(&network->momentum, sizeof(float), 1, fp);
  fwrite(&network->version, sizeof(int), 1, fp);

  // Layer count (including input layer)
  uint8_t num_layers = network->num_layers + 1;
  fwrite(&num_layers, sizeof(uint8_t), 1, fp);

  // Layer sizes (input layer first)
  uint32_t input_size = network->layers[0].input_size;
  fwrite(&input_size, sizeof(uint32_t), 1, fp);

  for (int i = 0; i < network->num_layers; i++) {
    uint32_t size = network->layers[i].output_size;
    fwrite(&size, sizeof(uint32_t), 1, fp);
  }

  // Activation functions
  for (int i = 0; i < network->num_layers; i++) {
    ActivationType act_type = ACT_RELU; // default

    if (network->layers[i].activation == relu) {
      act_type = ACT_RELU;
    } else if (network->layers[i].activation == softmax_placeholder) {
      act_type = ACT_SOFTMAX;
    }

    fwrite(&act_type, sizeof(ActivationType), 1, fp);
  }

  // Weights and biases
  for (int l = 0; l < network->num_layers; l++) {
    Layer_t *layer = &network->layers[l];
    size_t weights_size = layer->input_size * layer->output_size;
    size_t biases_size = layer->output_size;

    if (fwrite(layer->weights, sizeof(float), weights_size, fp) !=
            weights_size ||
        fwrite(layer->biases, sizeof(float), biases_size, fp) != biases_size) {
      fclose(fp);
      return -1;
    }
  }

  fclose(fp);
  return 0;
}

NeuralNetwork_t *load_Network(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    perror("Failed to open file for reading");
    return NULL;
  }

  // Read and verify header
  uint8_t magic;
  uint32_t file_version;
  if (fread(&magic, sizeof(uint8_t), 1, fp) != 1 ||
      fread(&file_version, sizeof(uint32_t), 1, fp) != 1) {
    fprintf(stderr, "Failed to read file header\n");
    fclose(fp);
    return NULL;
  }

  if (magic != MAGIC_NUMBER) {
    fprintf(stderr, "Invalid file format (bad magic number)\n");
    fclose(fp);
    return NULL;
  }

  if (file_version > FILE_VERSION) {
    fprintf(stderr, "Unsupported file version: %d (max supported: %d)\n",
            file_version, FILE_VERSION);
    fclose(fp);
    return NULL;
  }

  // Read network configuration
  float learning_rate, momentum;
  int version;
  if (fread(&learning_rate, sizeof(float), 1, fp) != 1 ||
      fread(&momentum, sizeof(float), 1, fp) != 1 ||
      fread(&version, sizeof(int), 1, fp) != 1) {
    fprintf(stderr, "Failed to read network configuration\n");
    fclose(fp);
    return NULL;
  }

  // Read layer count
  uint8_t num_layers;
  if (fread(&num_layers, sizeof(uint8_t), 1, fp) != 1) {
    fprintf(stderr, "Failed to read layer count\n");
    fclose(fp);
    return NULL;
  }

  if (num_layers < 2) {
    fprintf(stderr, "Invalid network: must have at least 2 layers\n");
    fclose(fp);
    return NULL;
  }

  // Read layer sizes
  uint32_t *layer_sizes = malloc(num_layers * sizeof(uint32_t));
  if (!layer_sizes) {
    perror("Memory allocation failed for layer sizes");
    fclose(fp);
    return NULL;
  }

  for (int i = 0; i < num_layers; i++) {
    if (fread(&layer_sizes[i], sizeof(uint32_t), 1, fp) != 1) {
      fprintf(stderr, "Failed to read layer size %d\n", i);
      free(layer_sizes);
      fclose(fp);
      return NULL;
    }
  }

  // Read activation functions
  ActivationType *act_types = malloc((num_layers - 1) * sizeof(ActivationType));
  if (!act_types) {
    perror("Memory allocation failed for activation types");
    free(layer_sizes);
    fclose(fp);
    return NULL;
  }

  for (int i = 0; i < num_layers - 1; i++) {
    if (fread(&act_types[i], sizeof(ActivationType), 1, fp) != 1) {
      fprintf(stderr, "Failed to read activation type for layer %d\n", i);
      free(layer_sizes);
      free(act_types);
      fclose(fp);
      return NULL;
    }
  }

  // Create activation function arrays
  ActivationFunc *activations =
      malloc((num_layers - 1) * sizeof(ActivationFunc));
  ActivationDerivative *derivatives =
      malloc((num_layers - 1) * sizeof(ActivationDerivative));

  if (!activations || !derivatives) {
    perror("Memory allocation failed for activation functions");
    free(layer_sizes);
    free(act_types);
    free(activations);
    free(derivatives);
    fclose(fp);
    return NULL;
  }

  for (int i = 0; i < num_layers - 1; i++) {
    switch (act_types[i]) {
    case ACT_RELU:
      activations[i] = relu;
      derivatives[i] = relu_derivative;
      break;
    case ACT_SOFTMAX:
      activations[i] = softmax_placeholder;
      derivatives[i] = NULL; // Handled specially
      break;
    default:
      fprintf(stderr, "Unknown activation type %d\n", act_types[i]);
      free(layer_sizes);
      free(act_types);
      free(activations);
      free(derivatives);
      fclose(fp);
      return NULL;
    }
  }

  // Create network
  NeuralNetwork_t *net =
      create_network((int *)layer_sizes, num_layers, activations, derivatives,
                     learning_rate, momentum);

  free(layer_sizes);
  free(act_types);
  free(activations);
  free(derivatives);

  if (!net) {
    fclose(fp);
    return NULL;
  }
  net->version = version;

  // Read weights and biases
  for (int l = 0; l < net->num_layers; l++) {
    Layer_t *layer = &net->layers[l];
    size_t weights_size = layer->input_size * layer->output_size;
    size_t biases_size = layer->output_size;

    if (fread(layer->weights, sizeof(float), weights_size, fp) !=
            weights_size ||
        fread(layer->biases, sizeof(float), biases_size, fp) != biases_size) {
      fprintf(stderr, "Failed to read layer %d parameters\n", l);
      free_network(net);
      fclose(fp);
      return NULL;
    }
  }

  fclose(fp);
  return net;
}
