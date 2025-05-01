
#include "nnlib-cl.h"

// Activation functions
float relu(float x) { return x > 0 ? x : 0; }
float relu_derivative(float x) { return x > 0 ? 1 : 0; }
float linear(float x) { return x; }
float linear_derivative(float x) { return 1; }
float softmax_placeholder(float x) { return x; }

void softmax(float *array, int size) {
  float max_val = array[0];
  for (int i = 1; i < size; i++) {
    if (array[i] > max_val)
      max_val = array[i];
  }

  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    array[i] = expf(array[i] - max_val);
    sum += array[i];
  }

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

// Network initialization
NeuralNetwork_t *create_network(int *layer_sizes, int num_layers,
                                ActivationFunc *activations,
                                ActivationDerivative *derivatives,
                                float learning_rate, float momentum,
                                cl_context context) {
  NeuralNetwork_t *net = (NeuralNetwork_t *)malloc(sizeof(NeuralNetwork_t));
  net->num_layers = num_layers - 1;
  net->layers = (Layer_t *)malloc(net->num_layers * sizeof(Layer_t));
  net->learning_rate = learning_rate;
  net->momentum = momentum;
  net->context = context;

  for (int i = 0; i < net->num_layers; i++) {
    Layer_t *layer = &net->layers[i];
    int input_size = layer_sizes[i];
    int output_size = layer_sizes[i + 1];
    int weights_size = input_size * output_size;

    layer->weights = (float *)malloc(weights_size * sizeof(float));
    layer->biases = (float *)calloc(output_size, sizeof(float));
    layer->weight_momentum = (float *)calloc(weights_size, sizeof(float));
    layer->bias_momentum = (float *)calloc(output_size, sizeof(float));
    layer->output = (float *)calloc(output_size, sizeof(float));
    layer->input = (float *)calloc(input_size, sizeof(float));

    // Initialize device buffers
    cl_int err;
    layer->weights_buf = clCreateBuffer(
        context, CL_MEM_READ_WRITE, weights_size * sizeof(float), NULL, &err);
    layer->biases_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       output_size * sizeof(float), NULL, &err);
    layer->weight_momentum_buf = clCreateBuffer(
        context, CL_MEM_READ_WRITE, weights_size * sizeof(float), NULL, &err);
    layer->bias_momentum_buf = clCreateBuffer(
        context, CL_MEM_READ_WRITE, output_size * sizeof(float), NULL, &err);
    layer->output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       output_size * sizeof(float), NULL, &err);
    layer->input_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      input_size * sizeof(float), NULL, &err);

    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation = activations[i];
    layer->activation_derivative = derivatives[i];

    // Xavier initialization
    float scale = sqrtf(2.0f / input_size);
    for (int j = 0; j < weights_size; j++) {
      layer->weights[j] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }
  }

  return net;
}

void free_network(NeuralNetwork_t *net) {
  for (int i = 0; i < net->num_layers; i++) {
    Layer_t *layer = &net->layers[i];

    free(layer->weights);
    free(layer->biases);
    free(layer->weight_momentum);
    free(layer->bias_momentum);
    free(layer->output);
    free(layer->input);

    clReleaseMemObject(layer->weights_buf);
    clReleaseMemObject(layer->biases_buf);
    clReleaseMemObject(layer->weight_momentum_buf);
    clReleaseMemObject(layer->bias_momentum_buf);
    clReleaseMemObject(layer->output_buf);
    clReleaseMemObject(layer->input_buf);
  }
  free(net->layers);
  free(net);
}

// Device memory management
void upload_network_to_device(NeuralNetwork_t *net, cl_command_queue queue) {
  cl_int err;
  for (int i = 0; i < net->num_layers; i++) {
    Layer_t *layer = &net->layers[i];
    int weights_size = layer->input_size * layer->output_size;

    err = clEnqueueWriteBuffer(queue, layer->weights_buf, CL_TRUE, 0,
                               weights_size * sizeof(float), layer->weights, 0,
                               NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, layer->biases_buf, CL_TRUE, 0,
                                layer->output_size * sizeof(float),
                                layer->biases, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, layer->weight_momentum_buf, CL_TRUE, 0,
                                weights_size * sizeof(float),
                                layer->weight_momentum, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, layer->bias_momentum_buf, CL_TRUE, 0,
                                layer->output_size * sizeof(float),
                                layer->bias_momentum, 0, NULL, NULL);

    if (err != CL_SUCCESS) {
      fprintf(stderr, "Error uploading layer %d to device\n", i);
    }
  }
}

void download_network_from_device(NeuralNetwork_t *net,
                                  cl_command_queue queue) {
  cl_int err;
  for (int i = 0; i < net->num_layers; i++) {
    Layer_t *layer = &net->layers[i];
    int weights_size = layer->input_size * layer->output_size;

    err = clEnqueueReadBuffer(queue, layer->weights_buf, CL_TRUE, 0,
                              weights_size * sizeof(float), layer->weights, 0,
                              NULL, NULL);
    err |= clEnqueueReadBuffer(queue, layer->biases_buf, CL_TRUE, 0,
                               layer->output_size * sizeof(float),
                               layer->biases, 0, NULL, NULL);

    if (err != CL_SUCCESS) {
      fprintf(stderr, "Error downloading layer %d from device\n", i);
    }
  }
}

// Forward pass with persistent buffers
float *forward_pass(NeuralNetwork_t *net, float *input, cl_command_queue queue,
                    cl_program program, int read_output) {
  cl_int err;
  cl_kernel forward_kernel = clCreateKernel(program, "forward_layer", &err);
  cl_kernel softmax_kernel = clCreateKernel(program, "softmax_layer", &err);

  // Upload input to first layer's input buffer
  Layer_t *first_layer = &net->layers[0];
  err = clEnqueueWriteBuffer(queue, first_layer->input_buf, CL_TRUE, 0,
                             first_layer->input_size * sizeof(float), input, 0,
                             NULL, NULL);

  for (int i = 0; i < net->num_layers; i++) {
    Layer_t *layer = &net->layers[i];

    // Set kernel arguments
    clSetKernelArg(forward_kernel, 0, sizeof(cl_mem), &layer->input_buf);
    clSetKernelArg(forward_kernel, 1, sizeof(cl_mem), &layer->weights_buf);
    clSetKernelArg(forward_kernel, 2, sizeof(cl_mem), &layer->biases_buf);
    clSetKernelArg(forward_kernel, 3, sizeof(cl_mem), &layer->output_buf);
    clSetKernelArg(forward_kernel, 4, sizeof(int), &layer->input_size);
    clSetKernelArg(forward_kernel, 5, sizeof(int), &layer->output_size);

    int act_type = (layer->activation == relu) ? 0 : 1;
    clSetKernelArg(forward_kernel, 6, sizeof(int), &act_type);

    // Execute kernel
    size_t global_size = layer->output_size;
    clEnqueueNDRangeKernel(queue, forward_kernel, 1, NULL, &global_size, NULL,
                           0, NULL, NULL);

    // Apply softmax if needed
    if (layer->activation == softmax_placeholder) {
      clSetKernelArg(softmax_kernel, 0, sizeof(cl_mem), &layer->output_buf);
      clSetKernelArg(softmax_kernel, 1, sizeof(cl_mem), &layer->output_buf);
      clSetKernelArg(softmax_kernel, 2, sizeof(int), &layer->output_size);

      size_t local_mem_size = 256 * sizeof(float);
      clSetKernelArg(softmax_kernel, 3, local_mem_size, NULL);
      clSetKernelArg(softmax_kernel, 4, local_mem_size, NULL);

      clEnqueueNDRangeKernel(queue, softmax_kernel, 1, NULL, &global_size, NULL,
                             0, NULL, NULL);
    }

    // For next layer, input is this layer's output
    if (i < net->num_layers - 1) {
      net->layers[i + 1].input_buf = layer->output_buf;
    }
  }

  // Read final output if requested
  Layer_t *output_layer = &net->layers[net->num_layers - 1];
  if (read_output) {
    clEnqueueReadBuffer(queue, output_layer->output_buf, CL_TRUE, 0,
                        output_layer->output_size * sizeof(float),
                        output_layer->output, 0, NULL, NULL);
    clFinish(queue);
  }

  clReleaseKernel(forward_kernel);
  clReleaseKernel(softmax_kernel);

  return output_layer->output;
}

// Backward pass with persistent buffers
void backward_pass(NeuralNetwork_t *net, float *target, cl_command_queue queue,
                   cl_program program) {
  cl_int err;
  cl_kernel output_back_kernel =
      clCreateKernel(program, "backward_output_layer", &err);
  cl_kernel hidden_back_kernel =
      clCreateKernel(program, "backward_hidden_layer", &err);
  cl_kernel update_kernel = clCreateKernel(program, "update_weights", &err);

  // Allocate delta buffers
  cl_mem *deltas_buffers = (cl_mem *)malloc(net->num_layers * sizeof(cl_mem));
  for (int i = 0; i < net->num_layers; i++) {
    deltas_buffers[i] =
        clCreateBuffer(net->context, CL_MEM_READ_WRITE,
                       net->layers[i].output_size * sizeof(float), NULL, &err);
  }

  // Output layer deltas
  Layer_t *output_layer = &net->layers[net->num_layers - 1];
  cl_mem target_buf =
      clCreateBuffer(net->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     output_layer->output_size * sizeof(float), target, &err);

  clSetKernelArg(output_back_kernel, 0, sizeof(cl_mem),
                 &output_layer->output_buf);
  clSetKernelArg(output_back_kernel, 1, sizeof(cl_mem), &target_buf);
  clSetKernelArg(output_back_kernel, 2, sizeof(cl_mem),
                 &deltas_buffers[net->num_layers - 1]);
  clSetKernelArg(output_back_kernel, 3, sizeof(int),
                 &output_layer->output_size);

  size_t global_size = output_layer->output_size;
  clEnqueueNDRangeKernel(queue, output_back_kernel, 1, NULL, &global_size, NULL,
                         0, NULL, NULL);

  // Backpropagate through hidden layers
  for (int l = net->num_layers - 2; l >= 0; l--) {
    Layer_t *current = &net->layers[l];
    Layer_t *next = &net->layers[l + 1];

    clSetKernelArg(hidden_back_kernel, 0, sizeof(cl_mem), &next->weights_buf);
    clSetKernelArg(hidden_back_kernel, 1, sizeof(cl_mem),
                   &deltas_buffers[l + 1]);
    clSetKernelArg(hidden_back_kernel, 2, sizeof(cl_mem), &current->input_buf);
    clSetKernelArg(hidden_back_kernel, 3, sizeof(cl_mem), &deltas_buffers[l]);
    clSetKernelArg(hidden_back_kernel, 4, sizeof(int), &current->output_size);
    clSetKernelArg(hidden_back_kernel, 5, sizeof(int), &next->output_size);

    global_size = current->output_size;
    clEnqueueNDRangeKernel(queue, hidden_back_kernel, 1, NULL, &global_size,
                           NULL, 0, NULL, NULL);
  }

  // Update weights and biases
  for (int l = 0; l < net->num_layers; l++) {
    Layer_t *layer = &net->layers[l];

    clSetKernelArg(update_kernel, 0, sizeof(cl_mem), &layer->input_buf);
    clSetKernelArg(update_kernel, 1, sizeof(cl_mem), &deltas_buffers[l]);
    clSetKernelArg(update_kernel, 2, sizeof(cl_mem), &layer->weights_buf);
    clSetKernelArg(update_kernel, 3, sizeof(cl_mem),
                   &layer->weight_momentum_buf);
    clSetKernelArg(update_kernel, 4, sizeof(cl_mem), &layer->biases_buf);
    clSetKernelArg(update_kernel, 5, sizeof(cl_mem), &layer->bias_momentum_buf);
    clSetKernelArg(update_kernel, 6, sizeof(float), &net->learning_rate);
    clSetKernelArg(update_kernel, 7, sizeof(float), &net->momentum);
    clSetKernelArg(update_kernel, 8, sizeof(int), &layer->input_size);
    clSetKernelArg(update_kernel, 9, sizeof(int), &layer->output_size);

    size_t global_dims[2] = {layer->input_size, layer->output_size};
    clEnqueueNDRangeKernel(queue, update_kernel, 2, NULL, global_dims, NULL, 0,
                           NULL, NULL);
  }

  // Cleanup
  clReleaseMemObject(target_buf);
  for (int i = 0; i < net->num_layers; i++) {
    clReleaseMemObject(deltas_buffers[i]);
  }
  free(deltas_buffers);

  clReleaseKernel(output_back_kernel);
  clReleaseKernel(hidden_back_kernel);
  clReleaseKernel(update_kernel);
}

// Training interface
void train(NeuralNetwork_t *net, float *input, float *target,
           cl_command_queue queue, cl_program program) {
  // Forward pass (don't read output back to host)
  forward_pass(net, input, queue, program, 0);

  // Backward pass
  backward_pass(net, target, queue, program);
}

int predict(NeuralNetwork_t *net, float *input, cl_command_queue queue,
            cl_program program) {
  // Forward pass and read output
  float *output = forward_pass(net, input, queue, program, 1);
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

// NeuralNetwork_t *load_Network(const char *filename) {
//   FILE *fp = fopen(filename, "rb");
//   if (!fp) {
//     perror("Failed to open file for reading");
//     return NULL;
//   }

//   // Read and verify header
//   uint8_t magic;
//   uint32_t file_version;
//   if (fread(&magic, sizeof(uint8_t), 1, fp) != 1 ||
//       fread(&file_version, sizeof(uint32_t), 1, fp) != 1) {
//     fprintf(stderr, "Failed to read file header\n");
//     fclose(fp);
//     return NULL;
//   }

//   if (magic != MAGIC_NUMBER) {
//     fprintf(stderr, "Invalid file format (bad magic number)\n");
//     fclose(fp);
//     return NULL;
//   }

//   if (file_version > FILE_VERSION) {
//     fprintf(stderr, "Unsupported file version: %d (max supported: %d)\n",
//             file_version, FILE_VERSION);
//     fclose(fp);
//     return NULL;
//   }

//   // Read network configuration
//   float learning_rate, momentum;
//   int version;
//   if (fread(&learning_rate, sizeof(float), 1, fp) != 1 ||
//       fread(&momentum, sizeof(float), 1, fp) != 1 ||
//       fread(&version, sizeof(int), 1, fp) != 1) {
//     fprintf(stderr, "Failed to read network configuration\n");
//     fclose(fp);
//     return NULL;
//   }

//   // Read layer count
//   uint8_t num_layers;
//   if (fread(&num_layers, sizeof(uint8_t), 1, fp) != 1) {
//     fprintf(stderr, "Failed to read layer count\n");
//     fclose(fp);
//     return NULL;
//   }

//   if (num_layers < 2) {
//     fprintf(stderr, "Invalid network: must have at least 2 layers\n");
//     fclose(fp);
//     return NULL;
//   }

//   // Read layer sizes
//   uint32_t *layer_sizes = malloc(num_layers * sizeof(uint32_t));
//   if (!layer_sizes) {
//     perror("Memory allocation failed for layer sizes");
//     fclose(fp);
//     return NULL;
//   }

//   for (int i = 0; i < num_layers; i++) {
//     if (fread(&layer_sizes[i], sizeof(uint32_t), 1, fp) != 1) {
//       fprintf(stderr, "Failed to read layer size %d\n", i);
//       free(layer_sizes);
//       fclose(fp);
//       return NULL;
//     }
//   }

//   // Read activation functions
//   ActivationType *act_types = malloc((num_layers - 1) * sizeof(ActivationType));
//   if (!act_types) {
//     perror("Memory allocation failed for activation types");
//     free(layer_sizes);
//     fclose(fp);
//     return NULL;
//   }

//   for (int i = 0; i < num_layers - 1; i++) {
//     if (fread(&act_types[i], sizeof(ActivationType), 1, fp) != 1) {
//       fprintf(stderr, "Failed to read activation type for layer %d\n", i);
//       free(layer_sizes);
//       free(act_types);
//       fclose(fp);
//       return NULL;
//     }
//   }

//   // Create activation function arrays
//   ActivationFunc *activations =
//       malloc((num_layers - 1) * sizeof(ActivationFunc));
//   ActivationDerivative *derivatives =
//       malloc((num_layers - 1) * sizeof(ActivationDerivative));

//   if (!activations || !derivatives) {
//     perror("Memory allocation failed for activation functions");
//     free(layer_sizes);
//     free(act_types);
//     free(activations);
//     free(derivatives);
//     fclose(fp);
//     return NULL;
//   }

//   for (int i = 0; i < num_layers - 1; i++) {
//     switch (act_types[i]) {
//     case ACT_RELU:
//       activations[i] = relu;
//       derivatives[i] = relu_derivative;
//       break;
//     case ACT_SOFTMAX:
//       activations[i] = softmax_placeholder;
//       derivatives[i] = NULL; // Handled specially
//       break;
//     default:
//       fprintf(stderr, "Unknown activation type %d\n", act_types[i]);
//       free(layer_sizes);
//       free(act_types);
//       free(activations);
//       free(derivatives);
//       fclose(fp);
//       return NULL;
//     }
//   }

//   // Create network
//   NeuralNetwork_t *net =
//       create_network((int *)layer_sizes, num_layers, activations, derivatives,
//                      learning_rate, momentum);

//   free(layer_sizes);
//   free(act_types);
//   free(activations);
//   free(derivatives);

//   if (!net) {
//     fclose(fp);
//     return NULL;
//   }
//   net->version = version;

//   // Read weights and biases
//   for (int l = 0; l < net->num_layers; l++) {
//     Layer_t *layer = &net->layers[l];
//     size_t weights_size = layer->input_size * layer->output_size;
//     size_t biases_size = layer->output_size;

//     if (fread(layer->weights, sizeof(float), weights_size, fp) !=
//             weights_size ||
//         fread(layer->biases, sizeof(float), biases_size, fp) != biases_size) {
//       fprintf(stderr, "Failed to read layer %d parameters\n", l);
//       free_network(net);
//       fclose(fp);
//       return NULL;
//     }
//   }

//   fclose(fp);
//   return net;
// }
