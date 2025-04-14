#include "nnlib.h"

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
  free(net);
}

float *forward_pass(NeuralNetwork_t *net, float *input) {
  float *current_input = input;

  for (int i = 0; i < net->num_layers; i++) {
    Layer_t *layer = &net->layers[i];

    // TODO parallelize this
    // Compute Wx + b
    for (int j = 0; j < layer->output_size; j++) {
      layer->input[j] = layer->biases[j];
      for (int k = 0; k < layer->input_size; k++) {
        layer->input[j] +=
            current_input[k] * layer->weights[k * layer->output_size + j];
      }
    }

    // ACTIVATION
    if (layer->activation == softmax_placeholder) {
      memcpy(layer->output, layer->input, layer->output_size * sizeof(float));
      softmax(layer->output, layer->output_size);
    } else {
      for (int j = 0; j < layer->output_size; j++) {
        layer->output[j] = layer->activation(layer->input[j]);
      }
    }
    current_input = layer->output;
  }

  return current_input;
}

void backward_pass(NeuralNetwork_t *net, float *input, float *target) {
  // Forward pass to store all intermediate values
  float *output = forward_pass(net, input);
  Layer_t *output_layer = &net->layers[net->num_layers - 1];

  // Allocate space for gradients
  float **deltas = (float **)malloc(net->num_layers * sizeof(float *));
  for (int i = 0; i < net->num_layers; i++) {
    deltas[i] = (float *)calloc(net->layers[i].output_size, sizeof(float));
  }

  // ACTIVATION
  // Output layer gradient
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

  // Backpropagate through hidden layers
  for (int l = net->num_layers - 2; l >= 0; l--) {
    Layer_t *current_layer = &net->layers[l];
    Layer_t *next_layer = &net->layers[l + 1];

    for (int i = 0; i < current_layer->output_size; i++) {
      float error = 0;
      for (int j = 0; j < next_layer->output_size; j++) {
        error += next_layer->weights[i * next_layer->output_size + j] *
                 deltas[l + 1][j];
      }
      deltas[l][i] =
          error * current_layer->activation_derivative(current_layer->input[i]);
    }
  }

  // Update weights and biases
  float *prev_output = input;
  for (int l = 0; l < net->num_layers; l++) {
    Layer_t *layer = &net->layers[l];

    // Update weights
    for (int i = 0; i < layer->input_size; i++) {
      for (int j = 0; j < layer->output_size; j++) {
        int idx = i * layer->output_size + j;
        float gradient = prev_output[i] * deltas[l][j];

        // Momentum update
        layer->weight_momentum[idx] =
            net->momentum * layer->weight_momentum[idx] +
            net->learning_rate * gradient;

        layer->weights[idx] -= layer->weight_momentum[idx];
      }
    }

    // Update biases
    for (int j = 0; j < layer->output_size; j++) {
      layer->bias_momentum[j] = net->momentum * layer->bias_momentum[j] +
                                net->learning_rate * deltas[l][j];

      layer->biases[j] -= layer->bias_momentum[j];
    }

    prev_output = layer->output;
  }

  // Free memory
  for (int i = 0; i < net->num_layers; i++) {
    free(deltas[i]);
  }
  free(deltas);
}

void train(NeuralNetwork_t *net, float *input, float *target) {
  backward_pass(net, input, target);
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

float calculate_loss(NeuralNetwork_t *net, float *output, float *target) {
  Layer_t *output_layer = &net->layers[net->num_layers - 1];
  float loss = 0;

  if (output_layer->activation == softmax_placeholder) {
    // Cross-entropy loss
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
