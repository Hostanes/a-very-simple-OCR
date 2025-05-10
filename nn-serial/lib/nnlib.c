#include "nnlib.h"

/*
  Forward Pass for a batch:

  calculating all A and Z values for
  each layer for every input in the batch in
  one operation.
  and stores them in the `neuron_Values' array
*/
void forward_Pass(float *batch, float *weights, float *biases,
                  float *neuron_Values, int batch_Size, int *layer_Sizes,
                  int num_Layers) {
  int weight_offset = 0;
  int bias_offset = 0;
  int neuron_value_offset = 0;

  for (int layer = 1; layer < num_Layers; layer++) {
    int input_size = layer_Sizes[layer - 1];
    int output_size = layer_Sizes[layer];
    int is_output_layer = (layer == num_Layers - 1);

    for (int sample = 0; sample < batch_Size; sample++) {

      // ===========================
      // --- Z value calculation ---
      // ===========================
      for (int neuron = 0; neuron < output_size; neuron++) {
        float Z = biases[bias_offset + neuron];

        for (int input_idx = 0; input_idx < input_size; input_idx++) {
          // first layer takes the input from the image in *batch,
          // else use the previous layers A values
          float input_val =
              (layer == 1)
                  ? batch[sample * input_size + input_idx]
                  : neuron_Values[(sample * (neuron_value_offset -
                                             layer_Sizes[layer - 1] * 2)) +
                                  (2 * input_idx) + 1];

          Z += input_val *
               weights[weight_offset + neuron * input_size + input_idx];
        }

        int current_offset = sample * (2 * output_size) + (2 * neuron);
        neuron_Values[neuron_value_offset + current_offset] = Z;

        // ===========================
        // --- Activation Function ---
        // ===========================
        // if ReLU layer use ReLU, instead store Z value as is to process layer
        if (!is_output_layer) {
          neuron_Values[neuron_value_offset + current_offset + 1] =
              fmaxf(0.0f, Z);
        } else {
          neuron_Values[neuron_value_offset + current_offset + 1] = Z;
        }
      }

      // ================================
      // --- Softmax for output layer ---
      // ================================
      if (is_output_layer) {
        // Find max for numerical stability
        float max_Z = -INFINITY;
        for (int neuron = 0; neuron < output_size; neuron++) {
          int offset = sample * (2 * output_size) + (2 * neuron);
          if (neuron_Values[neuron_value_offset + offset] > max_Z) {
            max_Z = neuron_Values[neuron_value_offset + offset];
          }
        }

        // Compute exp and exp sum
        float sum_exp = 0.0f;
        for (int neuron = 0; neuron < output_size; neuron++) {
          int offset = sample * (2 * output_size) + (2 * neuron);
          float exp_val =
              expf(neuron_Values[neuron_value_offset + offset] - max_Z);
          neuron_Values[neuron_value_offset + offset + 1] = exp_val;
          sum_exp += exp_val;
        }

        // Normalize to probabilities
        for (int neuron = 0; neuron < output_size; neuron++) {
          int offset = sample * (2 * output_size) + (2 * neuron) + 1;
          neuron_Values[neuron_value_offset + offset] /= sum_exp;
        }
      }
    }

    weight_offset += input_size * output_size;
    bias_offset += output_size;
    neuron_value_offset += 2 * output_size * batch_Size;
  }
}

/*
  Backward Pass (Gradient Calculation)

  Calculates gradients for weights and biases for all layers
  Acumulates gradients
  Stores results in gradient_weights and gradient_biases arrays
*/

void backward_Pass(float *batch, float *weights, float *neuron_Values,
                   float *gradient_weights, float *gradient_biases,
                   float *output_errors, int batch_Size, int *layer_Sizes,
                   int num_Layers) {
  // Initialize gradients to zero
  int total_weights = 0;
  int total_biases = 0;
  for (int i = 0; i < num_Layers - 1; i++) {
    total_weights += layer_Sizes[i] * layer_Sizes[i + 1];
    total_biases += layer_Sizes[i + 1];
  }

  memset(gradient_weights, 0, total_weights * sizeof(float));
  memset(gradient_biases, 0, total_biases * sizeof(float));

  // Find maximum layer size for error buffers
  int max_layer_size = 0;
  for (int i = 0; i < num_Layers; i++) {
    if (layer_Sizes[i] > max_layer_size) {
      max_layer_size = layer_Sizes[i];
    }
  }

  // Allocate error buffers
  float *errors = (float *)malloc(max_layer_size * sizeof(float));
  float *errors_next = (float *)malloc(max_layer_size * sizeof(float));

  // Initialize offsets
  int weight_offset = total_weights;
  int bias_offset = total_biases;
  int neuron_offset = 0;

  // Calculate initial neuron values offset
  for (int i = 1; i < num_Layers; i++) {
    neuron_offset += 2 * layer_Sizes[i] * batch_Size;
  }

  // Backward pass through layers
  for (int layer = num_Layers - 1; layer >= 1; layer--) {
    int input_size = layer_Sizes[layer - 1];
    int output_size = layer_Sizes[layer];

    // Update offsets for current layer
    weight_offset -= input_size * output_size;
    neuron_offset -= 2 * output_size * batch_Size;
    bias_offset -= output_size;

    for (int sample = 0; sample < batch_Size; sample++) {
      // Calculate errors for current layer
      for (int neuron = 0; neuron < output_size; neuron++) {
        int idx = neuron_offset + sample * (2 * output_size) + (2 * neuron);
        float Z = neuron_Values[idx];

        if (layer == num_Layers - 1) {
          // Output layer - use direct error
          errors[neuron] = output_errors[sample * output_size + neuron];
        } else {
          // Hidden layer - backpropagate error
          errors[neuron] = 0.0f;

          // Find the starting index of weights for next layer
          int next_weight_offset = 0;
          for (int l = 0; l < layer; l++) {
            next_weight_offset += layer_Sizes[l] * layer_Sizes[l + 1];
          }

          // Sum errors from next layer
          for (int next_neuron = 0; next_neuron < layer_Sizes[layer + 1];
               next_neuron++) {
            int weight_idx =
                next_weight_offset + next_neuron * output_size + neuron;
            errors[neuron] += errors_next[next_neuron] * weights[weight_idx];
          }

          // Apply ReLU derivative
          errors[neuron] *= (Z > 0) ? 1.0f : 0.0f;
        }
      }

      // Calculate gradients for current layer
      for (int neuron = 0; neuron < output_size; neuron++) {
        // Bias gradient
        gradient_biases[bias_offset + neuron] += errors[neuron];

        // Weight gradients
        for (int input_idx = 0; input_idx < input_size; input_idx++) {
          // Get input activation
          float input_activation;
          if (layer == 1) {
            // Input layer - get from batch
            input_activation = batch[sample * input_size + input_idx];
          } else {
            // Hidden layer - get from previous layer's activations
            int prev_idx = (neuron_offset - 2 * input_size * batch_Size) +
                           sample * (2 * input_size) + (2 * input_idx) + 1;
            input_activation = neuron_Values[prev_idx];
          }

          // Update weight gradient
          gradient_weights[weight_offset + neuron * input_size + input_idx] +=
              errors[neuron] * input_activation;
        }
      }

      // Store errors for next layer (only needed for hidden layers)
      if (layer > 1) {
        memcpy(errors_next, errors, output_size * sizeof(float));
      }
    }
  }

  // Average gradients over batch
  float inv_batch_size = 1.0f / batch_Size;
  for (int i = 0; i < total_weights; i++) {
    gradient_weights[i] *= inv_batch_size;
  }
  for (int i = 0; i < total_biases; i++) {
    gradient_biases[i] *= inv_batch_size;
  }

  // Clean up
  free(errors);
  free(errors_next);
}

/*
  Update Weights and Biases using Gradient Descent

  Updates the weights and biases arrays in-place using the calculated gradients
  and a specified learning rate.

  Parameters:
  - weights: Current weights array (flattened 1D)
  - biases: Current biases array (flattened 1D)
  - gradient_weights: Gradients for weights from backward pass
  - gradient_biases: Gradients for biases from backward pass
  - learning_rate: Step size for gradient descent
  - layer_Sizes: Array of layer sizes
  - num_Layers: Total number of layers in network
*/
void update_Weights(float *weights, float *biases,
                    const float *gradient_weights, const float *gradient_biases,
                    float learning_rate, const int *layer_Sizes,
                    int num_Layers) {
  // Calculate total number of weights and biases
  int total_weights = 0;
  int total_biases = 0;

  for (int i = 0; i < num_Layers - 1; i++) {
    total_weights += layer_Sizes[i] * layer_Sizes[i + 1];
    total_biases += layer_Sizes[i + 1];
  }

  // Update weights
  for (int i = 0; i < total_weights; i++) {
    weights[i] -= learning_rate * gradient_weights[i];
  }

  // Update biases
  for (int i = 0; i < total_biases; i++) {
    biases[i] -= learning_rate * gradient_biases[i];
  }
}
