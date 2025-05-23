#include "../lib/nnlib.h"
#include <stdio.h>

#define INPUT_SIZE 2
#define H1_SIZE 3
#define H2_SIZE 3
#define OUTPUT_SIZE 2
#define NUM_SAMPLES 3

int layer_sizes[] = {INPUT_SIZE, H1_SIZE, H2_SIZE, OUTPUT_SIZE};

// Total sizes (calculated manually)
#define TOTAL_WEIGHTS ((2 * 3) + (3 * 3) + (3 * 2)) // 6 + 9 + 6 = 21
#define TOTAL_BIASES (3 + 3 + 2)                    // = 8
#define TOTAL_Z (3 + 3 + 2)                         // = 8
#define TOTAL_A (2 + 3 + 3 + 2)                     // = 10

// Static arrays with fixed known values
float weights[TOTAL_WEIGHTS] = {
    // Weights: Layer 0 (2 -> 3)
    0.1f, 0.2f, // neuron 0
    0.3f, 0.4f, // neuron 1
    0.5f, 0.6f, // neuron 2

    // Weights: Layer 1 (3 -> 3)
    0.1f, 0.2f, 0.3f, //
    0.4f, 0.5f, 0.6f, //
    0.7f, 0.8f, 0.9f, //

    // Weights: Layer 2 (3 -> 2)
    0.1f, 0.2f, 0.3f, //
    0.4f, 0.5f, 0.6f};

float biases[TOTAL_BIASES] = {
    0.1f, 0.2f, 0.3f, // Layer 0

    0.4f, 0.5f, 0.6f, // Layer 1

    0.7f, 0.8f // Layer 2
};

float Z_values[TOTAL_Z];
float A_values[TOTAL_A];

// Offsets (computed manually for this specific architecture)
int weight_offsets[] = {0, 6, 15}; // Layer 0: 0, Layer 1: 6, Layer 2: 15

int activation_offsets[] = {0, 3, 6, 8};

int main() {
  float samples[NUM_SAMPLES][INPUT_SIZE] = {{1.0f, 2.0f},  //
                                            {0.5f, -1.0f}, //
                                            {2.0f, 0.0f}};

  int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);

  // One-hot encoded targets for 3 samples
  float targets[NUM_SAMPLES][OUTPUT_SIZE] = {
      {1.0f, 0.0f}, // target for sample 1
      {0.0f, 1.0f}, // target for sample 2
      {1.0f, 0.0f}  // target for sample 3
  };

  // Storage for gradients (reused per sample)
  float dZ_values[TOTAL_Z];
  float dW[TOTAL_WEIGHTS];
  float db[TOTAL_BIASES];
  float dA[TOTAL_A];

  float learning_rate = 0.01f;

  for (int epoch = 0; epoch < 2; ++epoch) {
    printf("\n\n================ Epoch %d ================\n", epoch + 1);
    for (int s = 0; s < NUM_SAMPLES; ++s) {
      printf("============ Sample %d ============\n", s + 1);

      // Forward pass
      forward_pass(samples[s], weights, biases, Z_values, A_values, layer_sizes,
                   num_layers, weight_offsets, activation_offsets);

      float *output = &A_values[activation_offsets[num_layers - 1]];
      for (int i = 0; i < OUTPUT_SIZE; ++i) {
        printf("  Output %d: %.4f\n", i, output[i]);
      }

      // Backward pass
      backward_pass(targets[s], weights, biases, Z_values, A_values, dZ_values,
                    dW, db, dA, layer_sizes, num_layers, weight_offsets,
                    activation_offsets);

      // Print gradients
      printf("\n  Gradients (dW):\n");
      for (int l = 0; l < num_layers - 1; ++l) {
        int rows = layer_sizes[l + 1];
        int cols = layer_sizes[l];
        int offset = weight_offsets[l];
        printf("    Layer %d (%d x %d):\n", l, rows, cols);
        for (int i = 0; i < rows; ++i) {
          printf("      ");
          for (int j = 0; j < cols; ++j) {
            printf("%.4f ", dW[offset + i * cols + j]);
          }
          printf("\n");
        }
      }

      printf("  Gradients (db):\n");
      for (int l = 0; l < num_layers - 1; ++l) {
        int size = layer_sizes[l + 1];
        int offset = activation_offsets[l];
        printf("    Layer %d:\n      ", l);
        for (int i = 0; i < size; ++i) {
          printf("%.4f ", db[offset + i]);
        }
        printf("\n");
      }

      // === Update weights and biases ===
      update_weights(weights, biases, dZ_values, A_values, layer_sizes,
                     num_layers, weight_offsets, activation_offsets,
                     learning_rate);

      // Print updated weights
      printf("  Updated Weights:\n");
      for (int l = 0; l < num_layers - 1; ++l) {
        int input_size = layer_sizes[l];
        int output_size = layer_sizes[l + 1];
        int offset = weight_offsets[l];

        printf("    Layer %d (%d inputs -> %d outputs):\n", l, input_size,
               output_size);
        for (int neuron = 0; neuron < output_size; ++neuron) {
          printf("      Neuron %d weights: ", neuron);
          for (int i = 0; i < input_size; ++i) {
            int index = offset + neuron * input_size + i;
            printf("%.4f ", weights[index]);
          }
          printf("\n");
        }
      }

      printf("\n");
    }
  }

  return 0;
}
