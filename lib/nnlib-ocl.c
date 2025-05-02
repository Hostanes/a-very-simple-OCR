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
      // FORWARD PASS
      write input to first_layer.input
      for each layer i:
          call forward kernel (ReLU or Softmax based on layer index)
          set next_layer.input = layer[i].activation

      // BACKWARD PASS
      for each layer i in reverse:
          call backward kernel (ReLU or Softmax based on layer index)
          optionally: compute gradient for previous layer if needed
    }
*/
