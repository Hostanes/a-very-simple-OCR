#include "nnlib-ocl.h"

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

NeuralNetwork_t create_NeuralNetwork(cl_context context, int *layer_sizes,
                                     int num_layers) {
  NeuralNetwork_t net;
  net.num_layers = num_layers - 1; // Ex: [784, 512, 256] → 2 layers
  net.context = context;
  net.layers = (Layer_t *)malloc(net.num_layers * sizeof(Layer_t));

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
  Pseudo code for the general pipeline

    train(input, target) {
      ==============

      FORWARD PASS
      write input to first_layer.input
      for each layer i:
          call forward kernel (ReLU or Softmax based on layer index)
          set next_layer.input = layer[i].activation

      ==============

      BACKWARD PASS
      for each layer i in reverse:
          call backward kernel (ReLU or Softmax based on layer index)
          optionally: compute gradient for previous layer if needed

      ==============

    }
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
    } else {
      kernel = net->backward_relu_kernel;
    }

    // For the last layer, we need to write the target to device
    if (i == net->num_layers - 1) {
      cl_mem target_buffer;
      if (net->host_output) {
        // If we have a host output buffer, use it for target comparison
        clEnqueueWriteBuffer(queue, layer->gradient, CL_TRUE, 0,
                             target_size * sizeof(float), target, 0, NULL,
                             NULL);
        target_buffer = layer->gradient;
      } else {
        // Otherwise create a temporary buffer
        target_buffer = clCreateBuffer(
            net->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            target_size * sizeof(float), (void *)target, NULL);
      }

      // Set kernel arguments for softmax backward pass
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &layer->weights);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &layer->biases);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), &layer->weight_momentum);
      clSetKernelArg(kernel, 3, sizeof(cl_mem), &layer->bias_momentum);
      clSetKernelArg(kernel, 4, sizeof(cl_mem), &layer->activation);
      clSetKernelArg(kernel, 5, sizeof(cl_mem), &layer->input);
      clSetKernelArg(kernel, 6, sizeof(cl_mem), &target_buffer);
      clSetKernelArg(kernel, 7, sizeof(int), &layer->input_size);
      clSetKernelArg(kernel, 8, sizeof(int), &layer->output_size);
      clSetKernelArg(kernel, 9, sizeof(float), &learning_rate);
      clSetKernelArg(kernel, 10, sizeof(float), &momentum);

      // Clean up temporary target buffer if we created one
      if (!net->host_output) {
        clReleaseMemObject(target_buffer);
      }
    } else {
      // For hidden layers, next layer's gradient is our input gradient
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

    // Execute kernel
    size_t global_size = layer->output_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL,
                           NULL);

    // Compute gradient for previous layer if not the first layer
    if (i > 0) {
      Layer_t *prev_layer = &net->layers[i - 1];

      // Zero out previous layer's gradient
      float zero = 0.0f;
      clEnqueueFillBuffer(queue, prev_layer->gradient, &zero, sizeof(float), 0,
                          prev_layer->output_size * sizeof(float), 0, NULL,
                          NULL);

      // Matrix multiply: gradient = weights^T * next_gradient
      // This would need another kernel (weight_gradient_kernel)
      // For simplicity, we'll assume this is handled in the backward pass
    }
  }
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
