#include "../lib/nnlib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define FLOAT_TOLERANCE 1e-3

int layer_Sizes[] = {2, 3, 2}; // Input(2) -> Hidden(3) -> Output(2)

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

void test_network() {
  printf("==== Neural Network Gradient Test ====\n");
  int total_errors = 0;
  const int num_Layers = 3;
  const int batch_Size = 2;

  // Test data - 2 samples with one-hot targets
  float batch[] = {0.5f, 1.0f,    // Sample 1
                   1.5f, 2.0f};   // Sample 2
  float targets[] = {1.0f, 0.0f,  // Sample 1 target (class 0)
                     0.0f, 1.0f}; // Sample 2 target (class 1)

  // Network parameters
  float weights[] = {
      // Layer 1 (2×3)
      0.1f, 0.2f, // Neuron 1
      0.3f, 0.4f, // Neuron 2
      0.5f, 0.6f, // Neuron 3
      // Layer 2 (3×2)
      1.0f, 1.1f, 1.2f, // Neuron 1
      1.3f, 1.4f, 1.5f  // Neuron 2
  };
  float biases[] = {0.1f, 0.2f, 0.3f, // Layer 1
                    0.4f, 0.5f};      // Layer 2

  // Allocate buffers
  int neuron_values_size = 0;
  for (int i = 1; i < num_Layers; i++) {
    neuron_values_size += layer_Sizes[i] * 2 * batch_Size;
  }
  float *neuron_values = (float *)calloc(neuron_values_size, sizeof(float));

  int total_weights =
      layer_Sizes[0] * layer_Sizes[1] + layer_Sizes[1] * layer_Sizes[2];
  int total_biases = layer_Sizes[1] + layer_Sizes[2];
  float *gradient_weights = (float *)calloc(total_weights, sizeof(float));
  float *gradient_biases = (float *)calloc(total_biases, sizeof(float));
  float *output_errors =
      (float *)malloc(layer_Sizes[2] * batch_Size * sizeof(float));

  // --- Forward Pass ---
  printf("\n=== Forward Pass ===\n");
  forward_Pass(batch, weights, biases, neuron_values, batch_Size, layer_Sizes,
               num_Layers);

  // Calculate output errors (prediction - target)
  for (int sample = 0; sample < batch_Size; sample++) {
    int output_offset =
        2 * layer_Sizes[1] * batch_Size + sample * (2 * layer_Sizes[2]);
    for (int neuron = 0; neuron < layer_Sizes[2]; neuron++) {
      output_errors[sample * layer_Sizes[2] + neuron] =
          neuron_values[output_offset + 2 * neuron + 1] -
          targets[sample * layer_Sizes[2] + neuron];
    }
  }

  // Print predictions vs targets
  for (int sample = 0; sample < batch_Size; sample++) {
    printf("\nSample %d:\n", sample + 1);
    printf("Target: [%.1f, %.1f]\n", targets[sample * 2],
           targets[sample * 2 + 1]);

    int out_offset =
        2 * layer_Sizes[1] * batch_Size + sample * (2 * layer_Sizes[2]);
    printf("Output: [%.4f, %.4f]\n", neuron_values[out_offset + 1],
           neuron_values[out_offset + 3]);

    printf("Errors: [%.4f, %.4f]\n", output_errors[sample * 2],
           output_errors[sample * 2 + 1]);
  }

  // --- Backward Pass ---
  printf("\n=== Backward Pass ===\n");
  backward_Pass(batch, weights, neuron_values, gradient_weights,
                gradient_biases, output_errors, batch_Size, layer_Sizes,
                num_Layers);

  // Expected gradients (manually calculated)
  // Layer 2 gradients (averaged over batch)
  float expected_grad_w2[] = {-0.055675f, -0.113675f, -0.171675f,
                              0.055675f,  0.113675f,  0.171675f};
  float expected_grad_b2[] = {-0.2435f, 0.2435f};

  // Layer 1 gradients (averaged)
  float expected_grad_w1[] = {0.006975f, 0.0435f,   0.006975f,
                              0.0435f,   0.006975f, 0.0435f};
  float expected_grad_b1[] = {0.07305f, 0.07305f, 0.07305f};

  // Verify gradients
  printf("\nChecking Layer 2 Gradients:\n");
  int w2_offset = layer_Sizes[0] * layer_Sizes[1];
  for (int i = 0; i < 6; i++) {
    char label[32];
    sprintf(label, "Weight gradient %d", i + 1);
    print_comparison(label, expected_grad_w2[i],
                     gradient_weights[w2_offset + i], &total_errors);
  }
  for (int i = 0; i < 2; i++) {
    char label[32];
    sprintf(label, "Bias gradient %d", i + 1);
    print_comparison(label, expected_grad_b2[i],
                     gradient_biases[layer_Sizes[1] + i], &total_errors);
  }

  printf("\nChecking Layer 1 Gradients:\n");
  for (int i = 0; i < 6; i++) {
    char label[32];
    sprintf(label, "Weight gradient %d", i + 1);
    print_comparison(label, expected_grad_w1[i], gradient_weights[i],
                     &total_errors);
  }
  for (int i = 0; i < 3; i++) {
    char label[32];
    sprintf(label, "Bias gradient %d", i + 1);
    print_comparison(label, expected_grad_b1[i], gradient_biases[i],
                     &total_errors);
  }

  // Cleanup
  free(neuron_values);
  free(gradient_weights);
  free(gradient_biases);
  free(output_errors);

  printf("\n==== Test Summary ====\n");
  if (total_errors == 0) {
    printf("✅ All gradients match expected values!\n");
  } else {
    printf("❌ Found %d gradient mismatches\n", total_errors);
  }
}

int main() {
  test_network();
  return 0;
}
