
/*
  version of nnlib.c that parallelizes each sample's operations
  parallel:
  - forward
  - backward
  Updated to use 2D arrays for weights and biases
*/

#include "nnlib.h"

#ifndef VERBOSE
#define printf(fmt, ...) (0)
#endif

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

// Helper function to allocate 2D float array
float **allocate_2d_array(int rows, int cols) {
  float **array = (float **)malloc(rows * sizeof(float *));
  if (!array)
    return NULL;

  for (int i = 0; i < rows; i++) {
    array[i] = (float *)calloc(cols, sizeof(float));
    if (!array[i]) {
      // Clean up on failure
      for (int j = 0; j < i; j++) {
        free(array[j]);
      }
      free(array);
      return NULL;
    }
  }
  return array;
}

// Helper function to free 2D float array
void free_2d_array(float **array, int rows) {
  if (!array)
    return;
  for (int i = 0; i < rows; i++) {
    free(array[i]);
  }
  free(array);
}

void initialize_layer(Layer_t *layer, int input_size, int output_size,
                      ActivationFunc activation,
                      ActivationDerivative derivative) {
  layer->input_size = input_size;
  layer->output_size = output_size;
  layer->activation = activation;
  layer->activation_derivative = derivative;

  // Allocate 2D arrays: weights[input_size][output_size]
  layer->weights = allocate_2d_array(input_size, output_size);
  layer->weight_momentum = allocate_2d_array(input_size, output_size);

  // Biases remain 1D as they're per output neuron
  layer->biases = (float *)calloc(output_size, sizeof(float));
  layer->bias_momentum = (float *)calloc(output_size, sizeof(float));
  layer->output = (float *)calloc(output_size, sizeof(float));
  layer->input = (float *)calloc(output_size, sizeof(float));

  // Check allocation success
  if (!layer->weights || !layer->weight_momentum || !layer->biases ||
      !layer->bias_momentum || !layer->output || !layer->input) {
    fprintf(stderr, "Memory allocation failed in initialize_layer\n");
    exit(1);
  }

  // Xavier initialization
  float scale = sqrtf(2.0f / input_size);
  randomize_weights_2d(layer->weights, input_size, output_size, scale);
}

void randomize_weights_2d(float **weights, int input_size, int output_size,
                          float scale) {
  for (int i = 0; i < input_size; i++) {
    for (int j = 0; j < output_size; j++) {
      weights[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }
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
    free_2d_array(net->layers[i].weights, net->layers[i].input_size);
    free_2d_array(net->layers[i].weight_momentum, net->layers[i].input_size);
    free(net->layers[i].biases);
    free(net->layers[i].bias_momentum);
    free(net->layers[i].output);
    free(net->layers[i].input);
  }
  free(net->layers);
  free(net);
}

float *forward_pass(NeuralNetwork_t *net, float *input) {
  printf("\n=== FORWARD PASS ===\n");
  float *current_input = input;

  for (int i = 0; i < net->num_layers; i++) {
    Layer_t *layer = &net->layers[i];
    printf("\nLayer %d (Input size: %d, Output size: %d)\n", i,
           layer->input_size, layer->output_size);

    // Print first 3 weights and biases for debugging
    printf("Weights[0][0:2]: [%.4f, %.4f, %.4f...]\n", layer->weights[0][0],
           layer->weights[0][1], layer->weights[0][2]);
    printf("Biases[0:2]: [%.4f, %.4f, %.4f...]\n", layer->biases[0],
           layer->biases[1], layer->biases[2]);

// Parallelized matrix multiplication
#pragma omp parallel for
    for (int j = 0; j < layer->output_size; j++) {
      float sum = layer->biases[j];
      for (int k = 0; k < layer->input_size; k++) {
        sum += current_input[k] * layer->weights[k][j];
      }
      layer->input[j] = sum;

      // Debug print for first few neurons
      if (j < 3) {
        printf("  Neuron %d pre-activation (z): %.4f\n", j, layer->input[j]);
      }
    }

    // Activation
    if (layer->activation == softmax_placeholder) {
      memcpy(layer->output, layer->input, layer->output_size * sizeof(float));
      softmax(layer->output, layer->output_size);
      printf("Softmax output[0:2]: [%.4f, %.4f, %.4f...]\n", layer->output[0],
             layer->output[1], layer->output[2]);
    } else {
#pragma omp parallel for
      for (int j = 0; j < layer->output_size; j++) {
        layer->output[j] = layer->activation(layer->input[j]);
        if (j < 3) {
          printf("  Neuron %d post-activation (a): %.4f\n", j,
                 layer->output[j]);
        }
      }
    }

    current_input = layer->output;
  }
  return current_input;
}

void backward_pass(NeuralNetwork_t *net, float *input, float *target) {
  printf("\n=== BACKWARD PASS ===\n");
  float *output = forward_pass(net, input);
  Layer_t *output_layer = &net->layers[net->num_layers - 1];

  printf("\n=== BACKWARD PASS ===\n");

  // Allocate and initialize deltas
  float **deltas = malloc(net->num_layers * sizeof(float *));
  for (int i = 0; i < net->num_layers; i++) {
    deltas[i] = calloc(net->layers[i].output_size, sizeof(float));
  }

  // Output layer error calculation
  printf("\nOutput Layer Errors:\n");
  if (output_layer->activation == softmax_placeholder) {
#pragma omp parallel for
    for (int i = 0; i < output_layer->output_size; i++) {
      deltas[net->num_layers - 1][i] = output_layer->output[i] - target[i];
      printf("  Output %d: (%.4f - %.4f) = %.4f\n", i, output_layer->output[i],
             target[i], deltas[net->num_layers - 1][i]);
    }
  } else {
#pragma omp parallel for
    for (int i = 0; i < output_layer->output_size; i++) {
      float error = output_layer->output[i] - target[i];
      float deriv = output_layer->activation_derivative(output_layer->input[i]);
      deltas[net->num_layers - 1][i] = error * deriv;
      printf("  Output %d: (%.4f - %.4f) * %.4f = %.4f\n", i,
             output_layer->output[i], target[i], deriv,
             deltas[net->num_layers - 1][i]);
    }
  }

  // Backpropagate errors
  for (int l = net->num_layers - 2; l >= 0; l--) {
    Layer_t *current = &net->layers[l];
    Layer_t *next = &net->layers[l + 1];
    printf("\nBackpropagating through layer %d\n", l);

#pragma omp parallel for
    for (int i = 0; i < current->output_size; i++) {
      float error = 0;
      for (int j = 0; j < next->output_size; j++) {
        error += next->weights[i][j] * deltas[l + 1][j];
      }
      float deriv = current->activation_derivative(current->input[i]);
      deltas[l][i] = error * deriv;

      if (i < 3) {
        printf("  Neuron %d: error=%.4f, deriv=%.4f, delta=%.4f\n", i, error,
               deriv, deltas[l][i]);
      }
    }
  }

  // Update weights and biases
  float *prev_output = input;
  for (int l = 0; l < net->num_layers; l++) {
    Layer_t *layer = &net->layers[l];
    printf("\nUpdating layer %d weights and biases\n", l);

    // Print pre-update values
    printf("Pre-update weights[0][0:2]: [%.4f, %.4f, %.4f...]\n",
           layer->weights[0][0], layer->weights[0][1], layer->weights[0][2]);
    printf("Pre-update biases[0:2]: [%.4f, %.4f, %.4f...]\n", layer->biases[0],
           layer->biases[1], layer->biases[2]);
    printf("Pre-update weight momentum[0][0:2]: [%.4f, %.4f, %.4f...]\n",
           layer->weight_momentum[0][0], layer->weight_momentum[0][1],
           layer->weight_momentum[0][2]);

// Update weights
#pragma omp parallel for
    for (int i = 0; i < layer->input_size; i++) {
      for (int j = 0; j < layer->output_size; j++) {
        float gradient = prev_output[i] * deltas[l][j];
        layer->weight_momentum[i][j] =
            net->momentum * layer->weight_momentum[i][j] +
            net->learning_rate * gradient;
        layer->weights[i][j] -= layer->weight_momentum[i][j];

        if (i == 0 && j < 3) {
          printf("  Weight update [%d][%d]: grad=%.4f, new_mom=%.4f, "
                 "new_weight=%.4f\n",
                 i, j, gradient, layer->weight_momentum[i][j],
                 layer->weights[i][j]);
        }
      }
    }

// Update biases
#pragma omp parallel for
    for (int j = 0; j < layer->output_size; j++) {
      layer->bias_momentum[j] = net->momentum * layer->bias_momentum[j] +
                                net->learning_rate * deltas[l][j];
      layer->biases[j] -= layer->bias_momentum[j];

      if (j < 3) {
        printf("  Bias update %d: delta=%.4f, new_mom=%.4f, new_bias=%.4f\n", j,
               deltas[l][j], layer->bias_momentum[j], layer->biases[j]);
      }
    }

    prev_output = layer->output;
  }

  // Clean up
  for (int i = 0; i < net->num_layers; i++) {
    free(deltas[i]);
  }
  free(deltas);
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
                       float ***gradients, float **bias_gradients) {
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
        error += next->weights[i][j] * deltas[l + 1][j];
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
        gradients[l][i][j] += prev_output[i] * deltas[l][j];
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
void apply_updates(NeuralNetwork_t *net, float ***gradients,
                   float **bias_gradients, int batch_size) {
  for (int l = 0; l < net->num_layers; l++) {
    Layer_t *layer = &net->layers[l];

    // Update weights with momentum
    for (int i = 0; i < layer->input_size; i++) {
      for (int j = 0; j < layer->output_size; j++) {
        float avg_gradient = gradients[l][i][j] / batch_size;
        layer->weight_momentum[i][j] =
            net->momentum * layer->weight_momentum[i][j] +
            net->learning_rate * avg_gradient;
        layer->weights[i][j] -= layer->weight_momentum[i][j];
      }
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
  Calculates loss of a layer depending on its activation function
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
