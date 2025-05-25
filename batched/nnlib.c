
#include "nnlib.h"

// Activation functions (same as before)
float relu(float x) { return x > 0 ? x : 0; }
float relu_derivative(float x) { return x > 0 ? 1 : 0; }
float tanh_activation(float x) { return tanhf(x); }
float tanh_derivative(float x) {
  float t = tanhf(x);
  return 1 - t * t;
}
float linear(float x) { return x; }
float linear_derivative(float x) { return 1; }
float softmax_placeholder(float x) { return x; }

// Batched softmax
void softmax_batch(float **arrays, int size, int batch_size) {
#pragma omp parallel for
  for (int b = 0; b < batch_size; b++) {
    // Find max value for numerical stability
    float max_val = arrays[b][0];
    for (int i = 1; i < size; i++) {
      if (arrays[b][i] > max_val) {
        max_val = arrays[b][i];
      }
    }

    // Calculate sum of exponentials
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
      arrays[b][i] = expf(arrays[b][i] - max_val);
      sum += arrays[b][i];
    }

    // Normalize
    for (int i = 0; i < size; i++) {
      arrays[b][i] /= sum;
    }
  }
}

// Batched softmax derivative
void softmax_derivative_batch(float **outputs, float **gradients, int size,
                              int batch_size) {
#pragma omp parallel for
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < size; i++) {
      gradients[b][i] = 0;
      for (int j = 0; j < size; j++) {
        float delta = (i == j) ? 1 : 0;
        gradients[b][i] +=
            outputs[b][i] * (delta - outputs[b][j]) * gradients[b][j];
      }
    }
  }
}

// Initialize layer with batch support
void initialize_layer(Layer_t *layer, int input_size, int output_size,
                      ActivationFunc activation,
                      ActivationDerivative derivative, int batch_size) {
  layer->input_size = input_size;
  layer->output_size = output_size;
  layer->batch_size = batch_size;
  layer->activation = activation;
  layer->activation_derivative = derivative;

  int weights_size = input_size * output_size;

  layer->weights = (float *)malloc(weights_size * sizeof(float));
  layer->biases = (float *)calloc(output_size, sizeof(float));
  layer->weight_momentum = (float *)calloc(weights_size, sizeof(float));
  layer->bias_momentum = (float *)calloc(output_size, sizeof(float));

  // Allocate batch storage
  layer->outputs = (float **)malloc(batch_size * sizeof(float *));
  layer->inputs = (float **)malloc(batch_size * sizeof(float *));
  for (int b = 0; b < batch_size; b++) {
    layer->outputs[b] = (float *)calloc(output_size, sizeof(float));
    layer->inputs[b] = (float *)calloc(output_size, sizeof(float));
  }

  // Xavier initialization
  float scale = sqrtf(2.0f / input_size);
  randomize_weights(layer->weights, weights_size, scale);
}

// Random weight initialization (same as before)
void randomize_weights(float *weights, int size, float scale) {
  for (int i = 0; i < size; i++) {
    weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
  }
}

// Create network with batch support
NeuralNetwork_t *create_network(int *layer_sizes, int num_layers,
                                ActivationFunc *activations,
                                ActivationDerivative *derivatives,
                                float learning_rate, float momentum,
                                int batch_size) {
  NeuralNetwork_t *net = (NeuralNetwork_t *)malloc(sizeof(NeuralNetwork_t));
  net->num_layers = num_layers - 1; // First size is input layer
  net->layers = (Layer_t *)malloc(net->num_layers * sizeof(Layer_t));
  net->learning_rate = learning_rate;
  net->momentum = momentum;
  net->batch_size = batch_size;

  for (int i = 0; i < net->num_layers; i++) {
    initialize_layer(&net->layers[i], layer_sizes[i], layer_sizes[i + 1],
                     activations[i], derivatives[i], batch_size);
  }

  return net;
}

// Free network with batch support
void free_network(NeuralNetwork_t *net) {
  for (int i = 0; i < net->num_layers; i++) {
    free(net->layers[i].weights);
    free(net->layers[i].biases);
    free(net->layers[i].weight_momentum);
    free(net->layers[i].bias_momentum);

    // Free batch storage
    for (int b = 0; b < net->layers[i].batch_size; b++) {
      free(net->layers[i].outputs[b]);
      free(net->layers[i].inputs[b]);
    }
    free(net->layers[i].outputs);
    free(net->layers[i].inputs);
  }
  free(net->layers);
  free(net);
}

// Resize network batch capacity
void resize_network_batch(NeuralNetwork_t *net, int new_batch_size) {
  if (new_batch_size == net->batch_size)
    return;

  for (int i = 0; i < net->num_layers; i++) {
    // Free old batch storage
    for (int b = 0; b < net->layers[i].batch_size; b++) {
      free(net->layers[i].outputs[b]);
      free(net->layers[i].inputs[b]);
    }
    free(net->layers[i].outputs);
    free(net->layers[i].inputs);

    // Allocate new batch storage
    net->layers[i].outputs = (float **)malloc(new_batch_size * sizeof(float *));
    net->layers[i].inputs = (float **)malloc(new_batch_size * sizeof(float *));
    for (int b = 0; b < new_batch_size; b++) {
      net->layers[i].outputs[b] =
          (float *)calloc(net->layers[i].output_size, sizeof(float));
      net->layers[i].inputs[b] =
          (float *)calloc(net->layers[i].output_size, sizeof(float));
    }

    net->layers[i].batch_size = new_batch_size;
  }

  net->batch_size = new_batch_size;
}

// Batched forward pass
float **forward_pass_batch(NeuralNetwork_t *net, float **inputs,
                           int batch_size) {
  if (batch_size != net->batch_size) {
    resize_network_batch(net, batch_size);
  }

  float **current_inputs = inputs;

  for (int l = 0; l < net->num_layers; l++) {
    Layer_t *layer = &net->layers[l];

// Parallelize across batch and output dimensions
#pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
      for (int j = 0; j < layer->output_size; j++) {
        float sum = layer->biases[j];
        for (int k = 0; k < layer->input_size; k++) {
          sum +=
              current_inputs[b][k] * layer->weights[k * layer->output_size + j];
        }
        layer->inputs[b][j] = sum;
      }
    }

    // Apply activation function
    if (layer->activation == softmax_placeholder) {
      // Softmax needs special handling
      for (int b = 0; b < batch_size; b++) {
        memcpy(layer->outputs[b], layer->inputs[b],
               layer->output_size * sizeof(float));
      }
      softmax_batch(layer->outputs, layer->output_size, batch_size);
    } else {
#pragma omp parallel for collapse(2)
      for (int b = 0; b < batch_size; b++) {
        for (int j = 0; j < layer->output_size; j++) {
          layer->outputs[b][j] = layer->activation(layer->inputs[b][j]);
        }
      }
    }

    current_inputs = layer->outputs;
  }

  return current_inputs;
}

// Batched backward pass
void backward_pass_batch(NeuralNetwork_t *net, float **inputs, float **targets,
                         int batch_size) {
  float **outputs = forward_pass_batch(net, inputs, batch_size);
  Layer_t *output_layer = &net->layers[net->num_layers - 1];

  // Allocate deltas for each sample in batch
  float ***deltas = (float ***)malloc(net->num_layers * sizeof(float **));
  for (int l = 0; l < net->num_layers; l++) {
    deltas[l] = (float **)malloc(batch_size * sizeof(float *));
    for (int b = 0; b < batch_size; b++) {
      deltas[l][b] = (float *)calloc(net->layers[l].output_size, sizeof(float));
    }
  }

  // Calculate output layer deltas
  if (output_layer->activation == softmax_placeholder) {
#pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
      for (int i = 0; i < output_layer->output_size; i++) {
        deltas[net->num_layers - 1][b][i] = outputs[b][i] - targets[b][i];
      }
    }
  } else {
#pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
      for (int i = 0; i < output_layer->output_size; i++) {
        float error = outputs[b][i] - targets[b][i];
        deltas[net->num_layers - 1][b][i] =
            error *
            output_layer->activation_derivative(output_layer->inputs[b][i]);
      }
    }
  }

  // Backpropagate through hidden layers
  for (int l = net->num_layers - 2; l >= 0; l--) {
    Layer_t *current = &net->layers[l];
    Layer_t *next = &net->layers[l + 1];

#pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
      for (int i = 0; i < current->output_size; i++) {
        float error = 0;
        for (int j = 0; j < next->output_size; j++) {
          error +=
              next->weights[i * next->output_size + j] * deltas[l + 1][b][j];
        }
        deltas[l][b][i] =
            error * current->activation_derivative(current->inputs[b][i]);
      }
    }
  }

  // Accumulate gradients across batch
  float **weight_gradients =
      (float **)malloc(net->num_layers * sizeof(float *));
  float **bias_gradients = (float **)malloc(net->num_layers * sizeof(float *));

  for (int l = 0; l < net->num_layers; l++) {
    Layer_t *layer = &net->layers[l];
    int weights_size = layer->input_size * layer->output_size;

    weight_gradients[l] = (float *)calloc(weights_size, sizeof(float));
    bias_gradients[l] = (float *)calloc(layer->output_size, sizeof(float));
  }

  float **prev_outputs = inputs;
  for (int l = 0; l < net->num_layers; l++) {
    Layer_t *layer = &net->layers[l];

// Accumulate weight gradients
#pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
      for (int i = 0; i < layer->input_size; i++) {
        for (int j = 0; j < layer->output_size; j++) {
          int idx = i * layer->output_size + j;
          weight_gradients[l][idx] += prev_outputs[b][i] * deltas[l][b][j];
        }
      }
    }

// Accumulate bias gradients
#pragma omp parallel for
    for (int j = 0; j < layer->output_size; j++) {
      for (int b = 0; b < batch_size; b++) {
        bias_gradients[l][j] += deltas[l][b][j];
      }
    }

    prev_outputs = layer->outputs;
  }

  // Apply updates with momentum
  for (int l = 0; l < net->num_layers; l++) {
    Layer_t *layer = &net->layers[l];
    int weights_size = layer->input_size * layer->output_size;

// Update weights
#pragma omp parallel for
    for (int i = 0; i < weights_size; i++) {
      float avg_gradient = weight_gradients[l][i] / batch_size;
      layer->weight_momentum[i] = net->momentum * layer->weight_momentum[i] +
                                  net->learning_rate * avg_gradient;
      layer->weights[i] -= layer->weight_momentum[i];
    }

// Update biases
#pragma omp parallel for
    for (int j = 0; j < layer->output_size; j++) {
      float avg_gradient = bias_gradients[l][j] / batch_size;
      layer->bias_momentum[j] = net->momentum * layer->bias_momentum[j] +
                                net->learning_rate * avg_gradient;
      layer->biases[j] -= layer->bias_momentum[j];
    }
  }

  // Clean up
  for (int l = 0; l < net->num_layers; l++) {
    free(weight_gradients[l]);
    free(bias_gradients[l]);
    for (int b = 0; b < batch_size; b++) {
      free(deltas[l][b]);
    }
    free(deltas[l]);
  }
  free(weight_gradients);
  free(bias_gradients);
  free(deltas);
}

// Batched training
void train_batch(NeuralNetwork_t *net, float **inputs, float **targets,
                 int batch_size) {
  backward_pass_batch(net, inputs, targets, batch_size);
}

// Batched prediction
int *predict_batch(NeuralNetwork_t *net, float **inputs, int batch_size) {
  float **outputs = forward_pass_batch(net, inputs, batch_size);
  Layer_t *output_layer = &net->layers[net->num_layers - 1];

  int *predictions = (int *)malloc(batch_size * sizeof(int));

#pragma omp parallel for
  for (int b = 0; b < batch_size; b++) {
    int max_index = 0;
    for (int i = 1; i < output_layer->output_size; i++) {
      if (outputs[b][i] > outputs[b][max_index]) {
        max_index = i;
      }
    }
    predictions[b] = max_index;
  }

  return predictions;
}

// Calculate loss for a batch
float calculate_batch_loss(NeuralNetwork_t *net, float **outputs,
                           float **targets, int batch_size) {
  Layer_t *output_layer = &net->layers[net->num_layers - 1];
  float total_loss = 0;

  if (output_layer->activation == softmax_placeholder) {
// Cross-entropy loss
#pragma omp parallel for reduction(+ : total_loss)
    for (int b = 0; b < batch_size; b++) {
      for (int i = 0; i < output_layer->output_size; i++) {
        if (targets[b][i] > 0) {
          total_loss += -targets[b][i] * logf(outputs[b][i] + 1e-10f);
        }
      }
    }
  } else {
// MSE loss
#pragma omp parallel for reduction(+ : total_loss)
    for (int b = 0; b < batch_size; b++) {
      for (int i = 0; i < output_layer->output_size; i++) {
        float diff = outputs[b][i] - targets[b][i];
        total_loss += diff * diff;
      }
    }
    total_loss /= (output_layer->output_size * batch_size);
  }

  return total_loss;
}
