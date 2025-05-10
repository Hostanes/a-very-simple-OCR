#include "../lib/nnlib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define FLOAT_TOLERANCE 1e-6

int layer_Sizes[] = {2, 3, 3, 2};

// Test data - 3 samples
float batch[] = {
    0.5f, 1.0f, // Sample 1
    1.5f, 2.0f, // Sample 2
    0.8f, 1.2f  // Sample 3
};

// Weights (manually constructed to test specific cases)
float weights[] = {
    // Layer 1 weights (2x3):
    0.1f, 0.2f, // Neuron 1 weights
    0.3f, 0.4f, // Neuron 2 weights
    0.5f, 0.6f, // Neuron 3 weights

    // Layer 2 (3x3 identity)
    1.0f, 0.0f, 0.0f, // Neuron 1
    0.0f, 1.0f, 0.0f, // Neuron 2
    0.0f, 0.0f, 1.0f, // Neuron 3

    // Layer 2 weights (3x2):
    1.0f, 1.1f, 1.2f, // Neuron 1 weights
    1.3f, 1.4f, 1.5f  // Neuron 2 weights
};

// Biases
float biases[] = {
    0.1f, 0.2f, 0.3f, // Layer 1 biases

    0.0f, 0.0f, 0.0f, // Layer 2 (identity)

    0.4f, 0.5f // Layer 2 biases
};

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

void test_forward_and_backward_pass() {
  printf("\n==== Starting forward + backward pass test ====\n");
  int total_errors = 0;

  int num_Layers = sizeof(layer_Sizes) / sizeof(layer_Sizes[0]);
  int batch_Size = 3;

  // Allocate space for neurons
  int neuron_Values_Size = 0;
  for (int i = 1; i < num_Layers; i++) {
    neuron_Values_Size += layer_Sizes[i] * 2 * batch_Size;
  }
  float *neuron_Values = (float *)calloc(neuron_Values_Size, sizeof(float));

  // Count number of weights
  int num_Of_Weights = 0;
  for (int i = 1; i < num_Layers; i++) {
    num_Of_Weights += layer_Sizes[i - 1] * layer_Sizes[i];
  }

  // Count number of biases
  int num_Of_Biases = 0;
  for (int i = 1; i < num_Layers; i++) {
    num_Of_Biases += layer_Sizes[i];
  }
  float *bias_Gradients = (float *)calloc(num_Of_Biases, sizeof(float));

  float *gradients = (float *)calloc(num_Of_Weights, sizeof(float));
  float *errors =
      (float *)calloc(layer_Sizes[num_Layers - 1] * batch_Size, sizeof(float));

  // One-hot encoded targets for 3 samples, output has 2 neurons
  float targets[] = {
      1.0f, 0.0f, // Sample 1
      0.0f, 1.0f, // Sample 2
      1.0f, 0.0f  // Sample 3
  };

  // Forward pass
  forward_Pass(batch, weights, biases, neuron_Values, batch_Size, layer_Sizes,
               num_Layers);

  // Backward pass
  backward_Pass(batch, weights, biases, neuron_Values, targets, gradients,
                bias_Gradients, errors, batch_Size, layer_Sizes, num_Layers,
                num_Of_Weights, neuron_Values_Size);

  // === Print Output Layer Activations (Softmax) ===
  printf("\n=== Softmax Output Layer (Layer %d) ===\n", num_Layers - 1);
  int output_layer_size = layer_Sizes[num_Layers - 1];
  int output_offset = neuron_Values_Size - (2 * output_layer_size * batch_Size);
  for (int i = 0; i < batch_Size; i++) {
    printf("Sample %d: ", i + 1);
    for (int j = 0; j < output_layer_size; j++) {
      float softmax_val =
          neuron_Values[output_offset + i * 2 * output_layer_size + 2 * j + 1];
      printf("%.6f ", softmax_val);
    }
    printf("\n");
  }

  // === Print Gradients ===
  printf("\n=== Gradients (dL/dW) ===\n");
  for (int i = 0; i < num_Of_Weights; i++) {
    printf("W[%d] grad = %.6f\n", i, gradients[i]);
  }
  printf("\n=== Gradients (dL/db) ===\n");
  for (int i = 0; i < num_Of_Biases; i++) {
    printf("b[%d] grad = %.6f\n", i, bias_Gradients[i]);
  }

  free(neuron_Values);
  free(errors);
  free(gradients);
}

int main() {
  test_forward_and_backward_pass();
  return 0;
}
