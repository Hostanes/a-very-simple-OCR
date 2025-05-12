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
              fmaxf(0.0f, Z);
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

  int last_hidden_size = layer_Sizes[num_Layers - 2];
  int output_size = layer_Sizes[num_Layers - 1];

  int weight_offset = num_Of_Weights - last_hidden_size * output_size;
  int value_offset = num_Of_Neurons - 2 * output_size * batch_Size;

  // ====================================
  // --- Output layer error (softmax) ---
  // ====================================
  printf("\n--- Output Layer Error ---\n");
  for (int sample = 0; sample < batch_Size; sample++) {
    for (int neuron = 0; neuron < output_size; neuron++) {
      int offset = sample * (2 * output_size) + (2 * neuron) + 1;
      float prediction = neuron_Values[value_offset + offset];
      float target = targets[sample * output_size + neuron];
      errors[value_offset + offset - 1] = prediction - target;
      printf("Sample %d, Neuron %d: Prediction = %.6f, Target = %.6f, Error "
             "(dLoss/dZ) = %.6f\n",
             sample + 1, neuron + 1, prediction, target,
             errors[value_offset + offset - 1]);
    }
  }

  // ========================
  // --- Backpropagation   ---
  // ========================
  printf("\n--- Backpropagation ---\n");
  for (int layer = num_Layers - 1; layer > 0; layer--) {
    int input_size = layer_Sizes[layer - 1];
    int output_size = layer_Sizes[layer];
    int is_output_layer = (layer == num_Layers - 1);

    // Calculate the starting index for the current layer's weights
    int layer_start_index = 0;
    for (int l = 1; l < layer; l++) {
      layer_start_index += layer_Sizes[l - 1] * layer_Sizes[l];
    }

    int bias_start_index = 0;
    for (int l = 1; l < layer; l++) {
      bias_start_index += layer_Sizes[l];
    }

    int value_offset_curr = (layer - 1) * 2 * output_size * batch_Size;
    int value_offset_prev =
        (layer - 2 >= 0) ? (layer - 2) * 2 * input_size * batch_Size : 0;

    printf("\nLayer %d:\n", layer);

    // TODO data parallelism here
    for (int sample = 0; sample < batch_Size; sample++) {
      printf("  Sample %d:\n", sample + 1);

      // TODO model paraallelism here
      for (int neuron = 0; neuron < output_size; neuron++) {
        int neuron_offset = sample * (2 * output_size) + (2 * neuron);
        float dLoss_dZ;

        printf("    Neuron %d:\n", neuron + 1);

        if (is_output_layer) {
          dLoss_dZ = errors[value_offset + neuron_offset];
          printf("      (Output Layer) dLoss_dZ = errors[%d] = %.6f\n",
                 value_offset + neuron_offset, dLoss_dZ);
        } else {
          // Sum over next layer's errors weighted by connection
          dLoss_dZ = 0.0f;
          printf("      Calculating dLoss_dZ (from next layer):\n");
          for (int k = 0; k < layer_Sizes[layer + 1]; k++) {
            int w_idx = weight_offset + k * output_size + neuron;
            int err_idx = (layer * 2 * layer_Sizes[layer + 1] * batch_Size) +
                          (sample * 2 * layer_Sizes[layer + 1]) + (2 * k);
            float error_next = errors[err_idx];
            float weight_val = weights[w_idx];
            dLoss_dZ += error_next * weight_val;
            printf("        k = %d, error_next[%d] = %.6f, weights[%d] = %.6f, "
                   "dLoss_dZ += %.6f\n",
                   k + 1, err_idx, error_next, w_idx, weight_val,
                   error_next * weight_val);
          }

          // Apply ReLU derivative
          float Z = neuron_Values[value_offset_curr + neuron_offset];
          float relu_derivative = (Z > 0) ? 1.0f : 0.0f;
          dLoss_dZ *= relu_derivative;
          printf(
              "      Z = %.6f, ReLU' = %.1f, dLoss_dZ (after ReLU') = %.6f\n",
              Z, relu_derivative, dLoss_dZ);
        }

        errors[value_offset_curr + neuron_offset] = dLoss_dZ;

        bias_Gradients[bias_start_index + neuron] += dLoss_dZ;
        printf("        errors_curr[%d] = %.6f, bias_gradients[%d] += %.6f "
               "(now %.6f)\n",
               value_offset_curr + neuron_offset, dLoss_dZ,
               bias_start_index + neuron, dLoss_dZ,
               bias_Gradients[bias_start_index + neuron]);

        // ====================================
        // --- Compute gradient wrt weights ---
        // ====================================
        printf("      Calculating weight gradients:\n");
        for (int i = 0; i < input_size; i++) {
          float input_val;
          if (layer == 1) {
            input_val = batch[sample * input_size + i];
            printf("        (Input Layer) input_val[%d] = %.6f\n",
                   sample * input_size + i, input_val);
          } else {
            input_val = neuron_Values[value_offset_prev +
                                      sample * 2 * input_size + 2 * i + 1];
            printf("        (Prev Layer A) input_val[%d] = %.6f\n",
                   value_offset_prev + sample * 2 * input_size + 2 * i + 1,
                   input_val);
          }
          float weight_Gradient = input_val * dLoss_dZ;
          weight_Gradients[layer_start_index + neuron * input_size + i] +=
              weight_Gradient;
          printf("        i = %d, input_val = %.6f, dLoss_dZ = %.6f, "
                 "weight_gradient = %.6f, weight_Gradients[%d] += %.6f (now "
                 "%.6f)\n",
                 i, input_val, dLoss_dZ, weight_Gradient,
                 layer_start_index + neuron * input_size + i, weight_Gradient,
                 weight_Gradients[layer_start_index + neuron * input_size + i]);
        }
      }
    }

    weight_offset -= input_size * output_size;
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
                    int num_Biases) {

  for (int i = 0; i < num_Weights; i++) {
    weights[i] -= learning_Rate * weight_Gradients[i];
  }

  for (int i = 0; i < num_Biases; i++) {
    biases[i] -= learning_Rate * bias_Gradients[i];
  }

  //
  //
}
