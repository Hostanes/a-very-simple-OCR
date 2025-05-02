#include "nnlib-ocl.h"
#include <stdio.h>

// Function to read kernel source from file
char *load_kernel_source(const char *filename) {
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Error: Failed to open kernel file %s\n", filename);
    return NULL;
  }

  // Get file size
  fseek(fp, 0, SEEK_END);
  long size = ftell(fp);
  rewind(fp);

  // Allocate buffer
  char *source = (char *)malloc(size + 1);
  if (!source) {
    fprintf(stderr, "Error: Failed to allocate memory for kernel source\n");
    fclose(fp);
    return NULL;
  }

  // Read file content
  size_t read_size = fread(source, 1, size, fp);
  if (read_size != size) {
    fprintf(stderr, "Error: Failed to read kernel source\n");
    free(source);
    fclose(fp);
    return NULL;
  }

  source[size] = '\0'; // Null-terminate
  fclose(fp);
  return source;
}
Layer_t create_Layer(cl_context context, int input_size, int output_size) {
  Layer_t layer;
  layer.input_size = input_size;
  layer.output_size = output_size;

  size_t weight_size = sizeof(float) * input_size * output_size;
  size_t bias_size = sizeof(float) * output_size;
  size_t act_size = sizeof(float) * output_size;
  size_t grad_size = sizeof(float) * output_size;
  size_t in_size = sizeof(float) * input_size;

  cl_int err;

  layer.weights =
      clCreateBuffer(context, CL_MEM_READ_WRITE, weight_size, NULL, &err);
  layer.biases =
      clCreateBuffer(context, CL_MEM_READ_WRITE, bias_size, NULL, &err);
  layer.weight_momentum =
      clCreateBuffer(context, CL_MEM_READ_WRITE, weight_size, NULL, &err);
  layer.bias_momentum =
      clCreateBuffer(context, CL_MEM_READ_WRITE, bias_size, NULL, &err);
  layer.input = clCreateBuffer(context, CL_MEM_READ_WRITE, in_size, NULL, &err);
  layer.activation =
      clCreateBuffer(context, CL_MEM_READ_WRITE, act_size, NULL, &err);
  layer.gradient =
      clCreateBuffer(context, CL_MEM_READ_WRITE, grad_size, NULL, &err);

  return layer;
}

void free_Layer(Layer_t *layer) {
  clReleaseMemObject(layer->weights);
  clReleaseMemObject(layer->biases);
  clReleaseMemObject(layer->weight_momentum);
  clReleaseMemObject(layer->bias_momentum);
  clReleaseMemObject(layer->input);
  clReleaseMemObject(layer->activation);
  clReleaseMemObject(layer->gradient);
}

NeuralNetwork_t create_NeuralNetwork(cl_context context, cl_device_id device,
                                     cl_command_queue queue, int *layer_sizes,
                                     int num_layers) {

  NeuralNetwork_t net;
  net.num_layers = num_layers - 1;
  net.context = context;
  net.device = device; // Store device if needed
  net.queue = queue;
  net.layers = (Layer_t *)malloc(net.num_layers * sizeof(Layer_t));

  cl_int err;
  char *kernel_source = load_kernel_source("lib/kernels.cl");
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&kernel_source, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error: Failed to create program from source: %d\n", err);
    free(kernel_source);
    // Handle error appropriately (e.g., return or exit)
  }

  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    // Get build log and print errors (as in your original code)
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    char *log = (char *)malloc(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log,
                          NULL);
    fprintf(stderr, "Error: Failed to build program:\n%s\n", log);
    free(log);
    free(kernel_source);
    clReleaseProgram(program);
    // Handle error appropriately
  }
  free(kernel_source);
  net.program = program;

  net.forward_relu_kernel = clCreateKernel(program, "forward_With_Relu", &err);
  net.forward_softmax_kernel =
      clCreateKernel(program, "forward_With_Softmax", &err);
  net.backward_relu_kernel =
      clCreateKernel(program, "backward_With_Relu", &err);
  net.backward_softmax_kernel =
      clCreateKernel(program, "backward_With_Softmax", &err);
  net.weight_gradient_kernel = clCreateKernel(program, "weight_gradient", &err);

  for (int i = 0; i < net.num_layers; i++) {
    net.layers[i] = create_Layer(context, layer_sizes[i], layer_sizes[i + 1]);
  }

  return net;
}

void free_NeuralNetwork(NeuralNetwork_t *net) {
  for (int i = 0; i < net->num_layers; i++) {
    free_Layer(&net->layers[i]);
  }
  free(net->layers);
}

/*
 * Pseudo code for the general pipeline
 *
 * train(input, target) {
 * ==============
 *
 * FORWARD PASS
 * write input to first_layer.input
 * for each layer i:
 * call forward kernel (ReLU or Softmax based on layer index)
 * set next_layer.input = layer[i].activation
 *
 * ==============
 *
 * BACKWARD PASS
 * for each layer i in reverse:
 * call backward kernel (ReLU or Softmax based on layer index)
 * optionally: compute gradient for previous layer if needed
 *
 * ==============
 *
 * }
 */

void forward(NeuralNetwork_t *net, cl_command_queue queue, const float *input,
             int input_size, float *output, int output_size) {

  // OPENCL_WRITE writes to first layer's input buffer
  Layer_t *first_layer = &net->layers[0];
  clEnqueueWriteBuffer(queue, first_layer->input, CL_TRUE, 0,
                       input_size * sizeof(float), input, 0, NULL, NULL);

  // Loop over the layers
  for (int i = 0; i < net->num_layers; i++) {
    Layer_t *layer = &net->layers[i];
    cl_kernel kernel;

    // use Softmax kernel for last layer
    // ReLU for others
    if (i == net->num_layers - 1) {
      kernel = net->forward_softmax_kernel;
    } else {
      kernel = net->forward_relu_kernel;
    }

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &layer->weights);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &layer->biases);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &layer->activation);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &layer->input);
    clSetKernelArg(kernel, 4, sizeof(int), &layer->input_size);
    clSetKernelArg(kernel, 5, sizeof(int), &layer->output_size);

    // Execute kernel
    size_t global_size = layer->output_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL,
                           NULL);

    // If not last layer, copy activation to next layer's input
    if (i < net->num_layers - 1) {
      Layer_t *next_layer = &net->layers[i + 1];
      clEnqueueCopyBuffer(queue, layer->activation, next_layer->input, 0, 0,
                          layer->output_size * sizeof(float), 0, NULL, NULL);
    }
  }

  // Read final output if requested
  if (output != NULL) {
    Layer_t *last_layer = &net->layers[net->num_layers - 1];
    clEnqueueReadBuffer(queue, last_layer->activation, CL_TRUE, 0,
                        output_size * sizeof(float), output, 0, NULL, NULL);
  }
}

void backward(NeuralNetwork_t *net, cl_command_queue queue, const float *target,
              int target_size, float learning_rate, float momentum) {
  // Start from the last layer and move backwards
  for (int i = net->num_layers - 1; i >= 0; i--) {
    Layer_t *layer = &net->layers[i];
    cl_kernel kernel;

    // Determine which kernel to use (Softmax for last layer, ReLU for others)
    if (i == net->num_layers - 1) {
      kernel = net->backward_softmax_kernel;

      // Write target to the gradient buffer of the last layer
      clEnqueueWriteBuffer(queue, layer->gradient, CL_TRUE, 0,
                           target_size * sizeof(float), target, 0, NULL, NULL);

      // Set kernel arguments for softmax backward pass
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &layer->weights);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &layer->biases);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), &layer->weight_momentum);
      clSetKernelArg(kernel, 3, sizeof(cl_mem), &layer->bias_momentum);
      clSetKernelArg(kernel, 4, sizeof(cl_mem), &layer->activation);
      clSetKernelArg(kernel, 5, sizeof(cl_mem), &layer->input);
      clSetKernelArg(kernel, 6, sizeof(cl_mem),
                     &layer->gradient); // Target is in gradient buffer
      clSetKernelArg(kernel, 7, sizeof(int), &layer->input_size);
      clSetKernelArg(kernel, 8, sizeof(int), &layer->output_size);
      clSetKernelArg(kernel, 9, sizeof(float), &learning_rate);
      clSetKernelArg(kernel, 10, sizeof(float), &momentum);
    } else {
      kernel = net->backward_relu_kernel;
      Layer_t *next_layer = &net->layers[i + 1];

      // Set kernel arguments for ReLU backward pass
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &layer->weights);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &layer->biases);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), &layer->weight_momentum);
      clSetKernelArg(kernel, 3, sizeof(cl_mem), &layer->bias_momentum);
      clSetKernelArg(kernel, 4, sizeof(cl_mem), &next_layer->gradient);
      clSetKernelArg(kernel, 5, sizeof(cl_mem), &layer->activation);
      clSetKernelArg(kernel, 6, sizeof(cl_mem), &layer->input);
      clSetKernelArg(kernel, 7, sizeof(int), &layer->input_size);
      clSetKernelArg(kernel, 8, sizeof(int), &layer->output_size);
      clSetKernelArg(kernel, 9, sizeof(float), &learning_rate);
      clSetKernelArg(kernel, 10, sizeof(float), &momentum);
    }

    // Execute the backward kernel for the current layer
    size_t global_size = layer->output_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL,
                           NULL);

    // Compute gradient for previous layer if not the first layer
    if (i > 0) {
      Layer_t *prev_layer = &net->layers[i - 1];
      Layer_t *current_layer = &net->layers[i];

      cl_kernel weight_grad_kernel = net->weight_gradient_kernel;
      clSetKernelArg(weight_grad_kernel, 0, sizeof(cl_mem),
                     &current_layer->weights);
      clSetKernelArg(weight_grad_kernel, 1, sizeof(cl_mem),
                     &current_layer->gradient);
      clSetKernelArg(weight_grad_kernel, 2, sizeof(cl_mem),
                     &prev_layer->gradient);
      clSetKernelArg(weight_grad_kernel, 3, sizeof(int),
                     &prev_layer->output_size); // Input size of current layer
      clSetKernelArg(
          weight_grad_kernel, 4, sizeof(int),
          &current_layer->output_size); // Output size of current layer

      size_t prev_global_size = prev_layer->output_size;
      clEnqueueNDRangeKernel(queue, weight_grad_kernel, 1, NULL,
                             &prev_global_size, NULL, 0, NULL, NULL);
    }
  }
  clFinish(queue); // Ensure all operations are complete
}

void train(NeuralNetwork_t *net, cl_command_queue queue, const float *input,
           int input_size, const float *target, int target_size,
           float learning_rate, float momentum) {

  // Forward pass
  forward(net, queue, input, input_size, NULL, 0);

  // Backward pass
  backward(net, queue, target, target_size, learning_rate, momentum);

  // Ensure all operations are complete
  clFinish(queue);
}
