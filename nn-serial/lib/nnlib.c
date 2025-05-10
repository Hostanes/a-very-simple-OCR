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
      printf("SAMPLE: %d\n", sample + 1);

      // Calculate the offset to this sample's data in previous layer
      int prev_layer_offset =
          (layer == 1)
              ? 0
              : (2 * layer_Sizes[layer - 1] * batch_Size * (layer - 2)) +
                    (sample * 2 * layer_Sizes[layer - 1]);

      // ===========================
      // --- Z value calculation ---
      // ===========================
      for (int neuron = 0; neuron < output_size; neuron++) {
        float Z = biases[bias_offset + neuron];

        for (int input_idx = 0; input_idx < input_size; input_idx++) {
          float input_val;

          if (layer == 1) {
            // First hidden layer takes input directly from batch
            input_val = batch[sample * input_size + input_idx];
          } else {
            // Subsequent layers take input from previous layer's A values
            input_val = neuron_Values[prev_layer_offset + (2 * input_idx) + 1];
          }

          Z += input_val *
               weights[weight_offset + neuron * input_size + input_idx];
        }

        int current_offset = sample * (2 * output_size) + (2 * neuron);
        neuron_Values[neuron_value_offset + current_offset] = Z;

        // ===========================
        // --- Activation Function ---
        // ===========================
        if (!is_output_layer) {
          // ReLU activation for hidden layers
          neuron_Values[neuron_value_offset + current_offset + 1] =
              fmaxf(0.0f, Z);
        } else {
          // Output layer - store Z directly (softmax will process it)
          neuron_Values[neuron_value_offset + current_offset + 1] = Z;
        }
        printf("neuron_Values: %f\n",
               neuron_Values[neuron_value_offset + current_offset]);
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
