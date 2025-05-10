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

