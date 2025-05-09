#include "../lib/nnlib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define FLOAT_TOLERANCE 1e-6

int layer_Sizes[] = {2, 3, 2};

void print_comparison(const char *label, float expected, float actual,
                      int *errors) {
  float diff = fabs(expected - actual);
  if (diff > FLOAT_TOLERANCE) {
    printf("❌ %s\n  Expected: %.6f\n  Actual:   %.6f\n  Diff:     %.6f\n",
           label, expected, actual, diff);
    (*errors)++;
  } else {
    printf("✅ %s\n  Value: %.6f (matches expected)\n", label, actual);
  }
}

void print_layer_values(float *values, int sample, int layer, int input_size,
                        int output_size, int batch_size) {
  printf("\nSample %d, Layer %d Values:\n", sample + 1, layer + 1);
  printf("----------------------------\n");

  int layer_offset = 0;
  for (int l = 1; l <= layer; l++) {
    layer_offset += layer_Sizes[l] * 2 * batch_size;
  }

  int sample_offset = sample * (2 * output_size);

  printf("%-8s %-8s %-8s\n", "Neuron", "Z", "A");
  printf("-------- -------- --------\n");

  for (int neuron = 0; neuron < output_size; neuron++) {
    int idx = layer_offset + sample_offset + (2 * neuron);
    printf("%-8d %-8.4f %-8.4f\n", neuron + 1,
           values[idx],    // Z value
           values[idx + 1] // A value
    );
  }
}

void test_forward_pass() {
  printf("==== Starting forward_pass unit tests ====\n");
  int total_errors = 0;

  // Tiny test network: 2 inputs -> 3 hidden -> 2 outputs
  int num_Layers = 3;
  int batch_Size = 2;

  // Test data - 2 samples
  float batch[] = {
      0.5f, 1.0f, // Sample 1
      1.5f, 2.0f  // Sample 2
  };

  // Weights (manually constructed to test specific cases)
  float weights[] = {
      // Layer 1 weights (2x3):
      0.1f, 0.2f, // Neuron 1 weights
      0.3f, 0.4f, // Neuron 2 weights
      0.5f, 0.6f, // Neuron 3 weights
      // Layer 2 weights (3x2):
      1.0f, 1.1f, 1.2f, // Neuron 1 weights
      1.3f, 1.4f, 1.5f  // Neuron 2 weights
  };

  // Biases
  float biases[] = {
      0.1f, 0.2f, 0.3f, // Layer 1 biases
      0.4f, 0.5f        // Layer 2 biases
  };

  // Allocate neuron values
  int neuron_Values_Size = 0;
  for (int i = 1; i < num_Layers; i++) {
    neuron_Values_Size += layer_Sizes[i] * 2 * batch_Size;
  }
  float *neuron_Values = (float *)calloc(neuron_Values_Size, sizeof(float));

  // Run forward pass
  forward_Pass(batch, weights, biases, neuron_Values, batch_Size, layer_Sizes,
               num_Layers);

  // Print all values for each sample
  for (int sample = 0; sample < batch_Size; sample++) {
    printf("\n═══════════════════════════════\n");
    printf("      SAMPLE %d DETAILED OUTPUT\n", sample + 1);
    printf("═══════════════════════════════\n");

    // Print input
    printf("\nInput Values:\n");
    for (int i = 0; i < layer_Sizes[0]; i++) {
      printf("Input %d: %.4f\n", i + 1, batch[sample * layer_Sizes[0] + i]);
    }

    // Print each layer's values
    int layer_offset = 0;
    for (int layer = 1; layer < num_Layers; layer++) {
      int input_size = layer_Sizes[layer - 1];
      int output_size = layer_Sizes[layer];

      print_layer_values(neuron_Values, sample, layer - 1, input_size,
                         output_size, batch_Size);

      // Print test results for specific neurons
      if (layer == 1) {
        printf("\nLayer 1 Verification:\n");
        float expected_z = (sample == 0) ? 0.35f : 2.25f;
        int neuron_idx = (sample == 0) ? 0 : (2 * 2 * 3 + 2 * 2);
        print_comparison(
            "  Neuron Z value", expected_z,
            neuron_Values[layer_offset + sample * (2 * output_size) +
                          (sample == 0 ? 0 : 4)],
            &total_errors);
        print_comparison(
            "  Neuron A value", expected_z,
            neuron_Values[layer_offset + sample * (2 * output_size) +
                          (sample == 0 ? 1 : 5)],
            &total_errors);
      }

      layer_offset += output_size * 2 * batch_Size;
    }
  }

  // Additional softmax verification
  printf("\n=== Softmax Final Verification ===\n");
  int output_offset = 2 * 3 * 2;
  for (int sample = 0; sample < batch_Size; sample++) {
    printf("\n-- Sample %d --\n", sample + 1);
    float sum = 0.0f;

    for (int neuron = 0; neuron < layer_Sizes[num_Layers - 1]; neuron++) {
      int idx = output_offset + sample * (2 * 2) + neuron * 2 + 1;
      sum += neuron_Values[idx];
      printf("Output Neuron %d Probability: %.6f\n", neuron + 1,
             neuron_Values[idx]);
    }

    printf("Probability Sum: %.6f\n", sum);
    if (fabs(sum - 1.0f) > FLOAT_TOLERANCE) {
      printf("❌ Softmax sum not equal to 1.0 (diff: %.6f)\n",
             fabs(sum - 1.0f));
      total_errors++;
    }
  }

  free(neuron_Values);

  printf("\n==== Test Summary ====\n");
  if (total_errors == 0) {
    printf("✅ All tests passed successfully!\n");
  } else {
    printf("❌ Found %d error(s) in implementation\n", total_errors);
  }
}

int main() {
  test_forward_pass();
  return 0;
}
