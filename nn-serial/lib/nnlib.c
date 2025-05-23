/*
  nnlib.c
  A serial implementation of a Batched Stochastic Gradient Descent
*/

#include "nnlib.h"

/*
  Forward Pass for a batch:

  calculating all A and Z values for
  each layer for every input in the batch in
  one operation.
  and stores them in the `neuron_Values' array
*/

#ifndef VERBOSE
#define printf(fmt, ...) (0)
#endif

#define LEAKY_RELU_ALPHA 0.01f

float leaky_ReLU(float x) { return (x > 0) ? x : LEAKY_RELU_ALPHA * x; }

void initialize_weights(float *weights, float *biases, int *layer_Sizes,
                        int num_Layers) {
  int weight_Index = 0;
  int bias_Index = 0;

  for (int l = 1; l < num_Layers; l++) {
    int fan_in = layer_Sizes[l - 1];
    int fan_out = layer_Sizes[l];

    // Replace your initialization with debug prints:
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    printf("Layer %d: fan_in=%d fan_out=%d limit=%.6f\n", l, fan_in, fan_out,
           limit);

    // For MNIST first layer (784->512), limit should be ~0.06
    // If it's orders of magnitude different, initialization is broken

    // #pragma omp parallel for
    for (int j = 0; j < fan_out; j++) {
      for (int i = 0; i < fan_in; i++) {
        float r = ((float)rand() / RAND_MAX);     // [0, 1]
        float weight = (2.0f * r - 1.0f) * limit; // [-limit, limit]
        weights[weight_Index++] = weight;
      }
      biases[bias_Index++] = 0.0f; // Biases set to 0
    }
  }
}

/*
  forward_Pass

  takes a `batch` of inputs, runs through each layer
  and calculates each Z and A value for each neuron
  for each sample in batch. Stored in `neuron_Values`

  neuron_Values array is a 1d array in this form:
  {
    neuron 1 layer 1,
    neuron 2 layer 1,
    neuron 3 layer 1,
    neuron 1 layer 2,
    ...
    neuron 2 layer 3
  }
*/
void forward_Pass(float *batch, float *weights, float *biases,
                  float *neuron_Values, int batch_Size, int *layer_Sizes,
                  int num_Layers) {
  int weight_offset = 0;
  int bias_offset = 0;
  int neuron_value_offset = 0;

  printf("\n--- Forward Pass ---\n");

  for (int layer = 1; layer < num_Layers; layer++) {
    int input_size = layer_Sizes[layer - 1];
    int output_size = layer_Sizes[layer];
    int is_output_layer = (layer == num_Layers - 1);

    printf("\nLayer %d:\n", layer);
    printf("  Input size: %d, Output size: %d\n", input_size, output_size);

    // TODO data parallelism here
    // #pragma omp parallel for
    for (int sample = 0; sample < batch_Size; sample++) {
      printf("  Sample %d:\n", sample + 1);

      // Calculate the offset to this sample's data in previous layer
      int prev_layer_offset =
          (layer == 1)
              ? 0
              : (2 * layer_Sizes[layer - 1] * batch_Size * (layer - 2)) +
                    (sample * 2 * layer_Sizes[layer - 1]);
      printf("    Previous layer offset: %d\n", prev_layer_offset);

      // ===========================
      // --- Z value calculation ---
      // ===========================

      // TODO model parallelism here,
      // might need to switch the sample
      // and layer for loops here in forward,
      // #pragma omp parallel for
      for (int neuron = 0; neuron < output_size; neuron++) {
        float Z = biases[bias_offset + neuron];
        printf("    Neuron %d:\n", neuron + 1);

        printf("      Bias: biases[%d] = %.6f\n", bias_offset + neuron,
               biases[bias_offset + neuron]);

        for (int input_idx = 0; input_idx < input_size; input_idx++) {
          float input_val;

          if (layer == 1) {
            // First hidden layer takes input directly from batch
            input_val = batch[sample * input_size + input_idx];
            printf("        Input from batch[%d]: %.6f\n",
                   sample * input_size + input_idx, input_val);
          } else {

            // Subsequent layers take input from previous layer's A values
            input_val = neuron_Values[prev_layer_offset + (2 * input_idx) + 1];
            printf("        Input from prev layer A[%d]: %.6f\n",
                   prev_layer_offset + (2 * input_idx) + 1, input_val);
          }

          float weight =
              weights[weight_offset + neuron * input_size + input_idx];
          Z += input_val * weight;
          printf(
              "        Weight: weights[%d] = %.6f, Z += %.6f * %.6f = %.6f\n",
              weight_offset + neuron * input_size + input_idx, weight,
              input_val, weight, input_val * weight);
        }

        int current_offset = sample * (2 * output_size) + (2 * neuron);
        neuron_Values[neuron_value_offset + current_offset] = Z;
        printf("      Z = %.6f, neuron_Values[%d] = %.6f\n", Z,
               neuron_value_offset + current_offset, Z);

        // ===========================
        // --- Activation Function ---
        // ===========================
        if (!is_output_layer) {
          // ReLU activation for hidden layers
          neuron_Values[neuron_value_offset + current_offset + 1] =
              leaky_ReLU(Z);
          printf("      ReLU(Z) = %.6f, neuron_Values[%d] = %.6f\n",
                 neuron_Values[neuron_value_offset + current_offset + 1],
                 neuron_value_offset + current_offset + 1,
                 neuron_Values[neuron_value_offset + current_offset + 1]);
        } else {
          // Output layer - store Z directly (softmax will process it)
          neuron_Values[neuron_value_offset + current_offset + 1] = Z;
          printf("      Output Z = %.6f, neuron_Values[%d] = %.6f\n", Z,
                 neuron_value_offset + current_offset + 1, Z);
        }
      }

      // ================================
      // --- Softmax for output layer ---
      // ================================
      if (is_output_layer) {
        printf("    --- Softmax ---\n");
        // Find max for numerical stability
        float max_Z = -INFINITY;
        for (int neuron = 0; neuron < output_size; neuron++) {
          int offset = sample * (2 * output_size) + (2 * neuron);
          if (neuron_Values[neuron_value_offset + offset] > max_Z) {
            max_Z = neuron_Values[neuron_value_offset + offset];
          }
        }
        printf("      Max Z = %.6f\n", max_Z);

        // Compute exp and exp sum
        float sum_exp = 0.0f;
        for (int neuron = 0; neuron < output_size; neuron++) {
          int offset = sample * (2 * output_size) + (2 * neuron);
          float exp_val =
              expf(neuron_Values[neuron_value_offset + offset] - max_Z);
          neuron_Values[neuron_value_offset + offset + 1] = exp_val;
          sum_exp += exp_val;
          printf("      Neuron %d: Z - max_Z = %.6f, exp = %.6f\n", neuron + 1,
                 neuron_Values[neuron_value_offset + offset] - max_Z, exp_val);
        }
        printf("      Sum of exps = %.6f\n", sum_exp);

        // Normalize to probabilities
        for (int neuron = 0; neuron < output_size; neuron++) {
          int offset = sample * (2 * output_size) + (2 * neuron) + 1;
          neuron_Values[neuron_value_offset + offset] /= sum_exp;
          printf("      Neuron %d: Probability = %.6f\n", neuron + 1,
                 neuron_Values[neuron_value_offset + offset]);
        }
      }
    }

    weight_offset += input_size * output_size;
    bias_offset += output_size;
    neuron_value_offset += 2 * output_size * batch_Size;
  }
}

/*
  backward_Pass

  Runs backwards in the network starting from output layer,
  Calculates gradients of each weight
  - `gradients`: weight gradients
  - `bias_gradients`: bias gradients
  - same size and shape as `biases` and `weights` arrays
*/
void backward_Pass(float *batch, float *weights, float *biases,
                   float *neuron_Values, float *targets,
                   float *weight_Gradients, float *bias_Gradients,
                   float *errors, int batch_Size, int *layer_Sizes,
                   int num_Layers, int num_Of_Weights, int num_Of_Neurons) {

  // Calculate total error size needed
  int errors_Size = 0;
  for (int i = 1; i < num_Layers; i++) {
    errors_Size += layer_Sizes[i] * batch_Size;
  }

  // Initialize error offset tracker
  int error_offset = 0;

  // ====================================
  // --- Output layer error (softmax) ---
  // ====================================
  int output_size = layer_Sizes[num_Layers - 1];
  int output_value_offset = num_Of_Neurons - 2 * output_size * batch_Size;

  printf("\n--- Output Layer Error ---\n");
  for (int sample = 0; sample < batch_Size; sample++) {
    for (int neuron = 0; neuron < output_size; neuron++) {
      int value_offset = sample * (2 * output_size) + (2 * neuron) + 1;
      float prediction = neuron_Values[output_value_offset + value_offset];
      float target = targets[sample * output_size + neuron];
      errors[error_offset + sample * output_size + neuron] =
          prediction - target;

      printf("Sample %d, Neuron %d: Prediction = %.6f, Target = %.6f, Error = "
             "%.6f\n",
             sample + 1, neuron + 1, prediction, target,
             errors[error_offset + sample * output_size + neuron]);
    }
  }
  error_offset += output_size * batch_Size;

  // ========================
  // --- Backpropagation  ---
  // ========================
  printf("\n--- Backpropagation ---\n");

  // Start from last hidden layer and move backwards
  for (int layer = num_Layers - 2; layer > 0; layer--) {
    int current_size = layer_Sizes[layer];
    int next_size = layer_Sizes[layer + 1];
    int prev_size = layer_Sizes[layer - 1];

    // Calculate weight and bias offsets for this layer
    int weight_offset = 0;
    int bias_offset = 0;
    for (int l = 1; l <= layer; l++) {
      weight_offset += layer_Sizes[l - 1] * layer_Sizes[l];
    }

    for (int l = 1; l < layer; l++) {
      bias_offset += layer_Sizes[l];
    }

    int value_offset_curr = 0; // Offset for current layer's Z values
    int value_offset_prev = 0; // Offset for previous layer's A values

    // Calculate value offsets
    for (int l = 1; l < layer; l++) {
      value_offset_curr += 2 * layer_Sizes[l] * batch_Size;
    }
    if (layer > 1) {
      for (int l = 1; l < layer - 1; l++) {
        value_offset_prev += 2 * layer_Sizes[l] * batch_Size;
      }
    }

    printf("\nLayer %d (size %d):\n", layer, current_size);

    for (int sample = 0; sample < batch_Size; sample++) {
      printf("  Sample %d:\n", sample + 1);

      for (int neuron = 0; neuron < current_size; neuron++) {
        printf("    Neuron %d:\n", neuron + 1);

        // Calculate dLoss/dZ for this neuron
        float dLoss_dZ = 0.0f;
        printf("      Calculating dLoss_dZ from next layer (size %d):\n",
               next_size);

        for (int k = 0; k < next_size; k++) {
          int next_layer_error_idx =
              (error_offset - next_size * batch_Size) + sample * next_size + k;
          int weight_idx = weight_offset + k * current_size + neuron;

          dLoss_dZ += errors[next_layer_error_idx] * weights[weight_idx];

          printf("        k=%d: error[%d]=%.6f * weight[%d]=%.6f => += %.6f\n",
                 k, next_layer_error_idx, errors[next_layer_error_idx],
                 weight_idx, weights[weight_idx],
                 errors[next_layer_error_idx] * weights[weight_idx]);
        }

        // Apply activation derivative (ReLU)
        int z_offset =
            value_offset_curr + sample * (2 * current_size) + (2 * neuron);
        float Z = neuron_Values[z_offset];
        float relu_derivative = (Z > 0) ? 1.0f : LEAKY_RELU_ALPHA;
        dLoss_dZ *= relu_derivative;

        printf("      Z=%.6f, ReLU'=%.1f => dLoss_dZ=%.6f\n", Z,
               relu_derivative, dLoss_dZ);

        // Store error for this neuron
        errors[error_offset + sample * current_size + neuron] = dLoss_dZ;
        printf("      Stored error[%d] = %.6f\n",
               error_offset + sample * current_size + neuron, dLoss_dZ);

        // Update bias gradient
        bias_Gradients[bias_offset + neuron] += dLoss_dZ;
        printf("      bias_gradients[%d] += %.6f (now %.6f)\n",
               bias_offset + neuron, dLoss_dZ,
               bias_Gradients[bias_offset + neuron]);

        // Update weight gradients
        printf("      Calculating weight gradients:\n");
        for (int i = 0; i < prev_size; i++) {
          float input_val;
          if (layer == 1) {
            input_val = batch[sample * prev_size + i];
            printf("        Input from batch[%d]: %.6f\n",
                   sample * prev_size + i, input_val);
          } else {
            input_val = neuron_Values[value_offset_prev +
                                      sample * (2 * prev_size) + (2 * i) + 1];
            printf("        Input from prev layer A[%d]: %.6f\n",
                   value_offset_prev + sample * (2 * prev_size) + (2 * i) + 1,
                   input_val);
          }

          float grad = input_val * dLoss_dZ;
          int grad_idx = weight_offset + neuron * prev_size + i;
          weight_Gradients[grad_idx] += grad;

          printf(
              "        weight_Gradients[%d] += %.6f * %.6f = %.6f (now %.6f)\n",
              grad_idx, input_val, dLoss_dZ, grad, weight_Gradients[grad_idx]);
        }
      }
    }

    error_offset += current_size * batch_Size;
  }
}

/*
  update_Weights

  pseudocode:
  for(i = 0; i < num_weights; i++)
    // weight and gradient weights same shape
    weight[i] -= learning_Rate * gradient_weight[i]

  for(i = 0; i < num_biases; i++)
    // biases and gradient biases same shape
    bias[i]-= learning_Rate * gradient_biases[i]
*/

void update_Weights(float *weights, float *biases, float *weight_Gradients,
                    float learning_Rate, float *bias_Gradients, int num_Weights,
                    int num_Biases, int batch_Size) {

  float inv_batch_size = 1.0f / batch_Size;

  for (int i = 0; i < num_Weights; i++) {
    weights[i] -= learning_Rate * (weight_Gradients[i] * inv_batch_size);
    weight_Gradients[i] = 0.0f; // Reset for next batch
  }

  for (int i = 0; i < num_Biases; i++) {
    biases[i] -= learning_Rate * (bias_Gradients[i] * inv_batch_size);
    bias_Gradients[i] = 0.0f; // Reset for next batch
  }
}

float compute_loss(float *neuron_Values, float *targets, int batch_Size,
                   int output_size, int num_Layers) {
  float loss = 0.0f;
  int value_offset = 2 * output_size * batch_Size * (num_Layers - 1);

  for (int sample = 0; sample < batch_Size; sample++) {
    for (int neuron = 0; neuron < output_size; neuron++) {
      int offset = sample * (2 * output_size) + (2 * neuron) + 1;
      float pred = neuron_Values[value_offset + offset];
      float target = targets[sample * output_size + neuron];
      loss += -target * logf(pred + 1e-8); // Cross-entropy
    }
  }
  return loss / batch_Size;
}

void gradient_check(float *batch, float *weights, float *biases,
                    float *neuron_Values, float *targets,
                    float *weight_Gradients, float *bias_Gradients,
                    float *errors, int batch_Size, int *layer_Sizes,
                    int num_Layers, int num_Of_Weights, int num_Of_Neurons) {

  float epsilon = 1e-4;
  int test_weight_index = 0; // Test first weight for simplicity

  // Analytic gradient (from backprop)
  backward_Pass(batch, weights, biases, neuron_Values, targets,
                weight_Gradients, bias_Gradients, errors, batch_Size,
                layer_Sizes, num_Layers, num_Of_Weights, num_Of_Neurons);
  float analytic_grad = weight_Gradients[test_weight_index];

  // Numeric gradient
  float original_weight = weights[test_weight_index];

  // f(x + ε)
  weights[test_weight_index] = original_weight + epsilon;
  forward_Pass(batch, weights, biases, neuron_Values, batch_Size, layer_Sizes,
               num_Layers);
  float loss_plus = compute_loss(neuron_Values, targets, batch_Size,
                                 layer_Sizes[num_Layers - 1], num_Layers);

  // f(x - ε)
  weights[test_weight_index] = original_weight - epsilon;
  forward_Pass(batch, weights, biases, neuron_Values, batch_Size, layer_Sizes,
               num_Layers);
  float loss_minus = compute_loss(neuron_Values, targets, batch_Size,
                                  layer_Sizes[num_Layers - 1], num_Layers);

  float numeric_grad = (loss_plus - loss_minus) / (2 * epsilon);

  printf("Gradient Check:\n");
  printf("Analytic: %.8f vs Numeric: %.8f\n", analytic_grad, numeric_grad);
  printf("Relative Error: %.8f\n", fabs(analytic_grad - numeric_grad) /
                                       fabs(analytic_grad + numeric_grad));

  // Reset weight
  weights[test_weight_index] = original_weight;
}
