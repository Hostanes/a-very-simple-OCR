
/*
  A test of the forward and backward operations on the DNN
  layer sizes:
  - 2 input values
  - ReLU H1: 3
  - ReLU H2: 3
  - Softmax Output: 2
  Uses 6 samples divided into 2 batches of 3 samples each
*/

#include "../lib/nnlib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define FLOAT_TOLERANCE 1e-6
#define LEARNING_RATE 0.01
#define BATCH_SIZE 3
#define NUM_SAMPLES 6
#define NUM_BATCHES (NUM_SAMPLES / BATCH_SIZE)

int layer_Sizes[] = {2, 3, 3, 2};

// Test data - 6 samples (2 batches of 3 samples each)
float all_samples[] = {
    // Batch 1
    0.5f, 1.0f, // Sample 1
    1.5f, 2.0f, // Sample 2
    0.8f, 1.2f, // Sample 3

    // Batch 2
    0.3f, 0.7f, // Sample 4
    1.2f, 1.8f, // Sample 5
    0.9f, 1.1f  // Sample 6
};

// One-hot encoded targets for 6 samples, output has 2 neurons
float all_targets[] = {
    // Batch 1
    1.0f, 0.0f, // Sample 1
    0.0f, 1.0f, // Sample 2
    1.0f, 0.0f, // Sample 3

    // Batch 2
    0.0f, 1.0f, // Sample 4
    1.0f, 0.0f, // Sample 5
    0.0f, 1.0f  // Sample 6
};

// Weights (manually constructed to test specific cases)
float initial_weights[] = {
    // Layer 1 weights (2x3) indices 0 -> 5:
    0.1f, 0.2f, // Neuron 1 weights
    0.3f, 0.4f, // Neuron 2 weights
    0.5f, 0.6f, // Neuron 3 weights

    // Layer 2 (3x3 identity) indices 6 -> 14:
    1.0f, 0.0f, 0.0f, // Neuron 1
    0.0f, 1.0f, 0.0f, // Neuron 2
    0.0f, 0.0f, 1.0f, // Neuron 3

    // Layer 3 weights (3x2) indices 15 -> 20:
    1.0f, 1.1f, 1.2f, // Neuron 1 weights
    1.3f, 1.4f, 1.5f  // Neuron 2 weights
};

// Biases
float initial_biases[] = {
    0.1f, 0.2f, 0.3f, // Layer 1 biases 0 -> 2

    0.0f, 0.0f, 0.0f, // Layer 2 (identity) 3 -> 5

    0.4f, 0.5f // Layer 3 biases -> 6, 7
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

void print_layer_values(float *values, int sample, int layer, int batch_size) {
  printf("\nSample %d, Layer %d Values:\n", sample + 1, layer + 1);
  printf("----------------------------\n");

  int layer_offset = 0;
  for (int l = 1; l <= layer; l++) {
    layer_offset += layer_Sizes[l] * 2 * batch_size;
  }

  int sample_offset = sample * (2 * layer_Sizes[layer]);

  printf("%-8s %-8s %-8s\n", "Neuron", "Z", "A");
  printf("-------- -------- --------\n");

  for (int neuron = 0; neuron < layer_Sizes[layer]; neuron++) {
    int idx = layer_offset + sample_offset + (2 * neuron);
    printf("%-8d %-8.4f %-8.4f\n", neuron + 1,
           values[idx],    // Z value
           values[idx + 1] // A value
    );
  }
}

void test_forward_and_backward_pass() {
  printf("\n==== Starting forward + backward pass test with %d batches ====\n",
         NUM_BATCHES);
  int total_errors = 0;

  int num_Layers = sizeof(layer_Sizes) / sizeof(layer_Sizes[0]);

  // Allocate space for neurons
  int neuron_Values_Size = 0;
  for (int i = 1; i < num_Layers; i++) {
    neuron_Values_Size += layer_Sizes[i] * 2 * BATCH_SIZE;
  }
  float *neuron_Values = (float *)calloc(neuron_Values_Size, sizeof(float));

  // Count number of weights
  int num_Of_Weights = 0;
  for (int i = 1; i < num_Layers; i++) {
    num_Of_Weights += layer_Sizes[i - 1] * layer_Sizes[i];
  }

  // Count number of biases
  int num_Of_Biases = 0, errors_Size = 0;
  for (int i = 1; i < num_Layers; i++) {
    num_Of_Biases += layer_Sizes[i];
    errors_Size += layer_Sizes[i] * BATCH_SIZE;
  }
  printf("error size = %d\n", errors_Size);

  // Make copies of initial weights and biases for each batch
  float *weights = (float *)malloc(num_Of_Weights * sizeof(float));
  float *biases = (float *)malloc(num_Of_Biases * sizeof(float));
  memcpy(weights, initial_weights, num_Of_Weights * sizeof(float));
  memcpy(biases, initial_biases, num_Of_Biases * sizeof(float));

  float *bias_Gradients = (float *)calloc(num_Of_Biases, sizeof(float));
  float *gradients = (float *)calloc(num_Of_Weights, sizeof(float));

  float *errors = malloc(errors_Size * sizeof(float));
  if (!errors) {
    fprintf(stderr, "Failed to allocate errors\n");
    exit(EXIT_FAILURE);
  }

  // Process each batch
  for (int batch_num = 0; batch_num < NUM_BATCHES; batch_num++) {
    printf("\n=== Processing Batch %d ===\n", batch_num + 1);

    // Get pointers to current batch data and targets
    float *batch_data = all_samples + batch_num * BATCH_SIZE * layer_Sizes[0];
    float *batch_targets =
        all_targets + batch_num * BATCH_SIZE * layer_Sizes[num_Layers - 1];

    // Forward pass
    forward_Pass(batch_data, weights, biases, neuron_Values, BATCH_SIZE,
                 layer_Sizes, num_Layers);

    // Print sample outputs from this batch
    printf("\n=== Softmax Output Layer (Layer %d) ===\n", num_Layers - 1);
    int output_layer_size = layer_Sizes[num_Layers - 1];
    int output_offset =
        neuron_Values_Size - (2 * output_layer_size * BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; i++) {
      printf("Sample %d: ", batch_num * BATCH_SIZE + i + 1);
      for (int j = 0; j < output_layer_size; j++) {
        float softmax_val =
            neuron_Values[output_offset + i * 2 * output_layer_size + 2 * j +
                          1];
        printf("%.6f ", softmax_val);
      }
      printf("\n");
    }

    // Backward pass
    backward_Pass(batch_data, weights, biases, neuron_Values, batch_targets,
                  gradients, bias_Gradients, errors, BATCH_SIZE, layer_Sizes,
                  num_Layers, num_Of_Weights, neuron_Values_Size);

    // Print gradients for this batch
    printf("\n=== Gradients (dL/dW) for Batch %d ===\n", batch_num + 1);
    for (int i = 0; i < num_Of_Weights; i++) {
      printf("W[%d] grad = %.6f\n", i, gradients[i]);
    }
    printf("\n=== Gradients (dL/db) for Batch %d ===\n", batch_num + 1);
    for (int i = 0; i < num_Of_Biases; i++) {
      printf("b[%d] grad = %.6f\n", i, bias_Gradients[i]);
    }

    // Update weights after processing this batch
    printf("\n=== Updating Weights after Batch %d ===\n", batch_num + 1);
    update_Weights(weights, biases, gradients, LEARNING_RATE, bias_Gradients,
                   num_Of_Weights, num_Of_Biases, BATCH_SIZE);

    printf("\nNew weights after batch %d:\n", batch_num + 1);
    for (int i = 0; i < num_Of_Weights; i++) {
      printf("weights[%d] = %f\n", i, weights[i]);
    }

    printf("\nNew biases after batch %d:\n", batch_num + 1);
    for (int i = 0; i < num_Of_Biases; i++) {
      printf("biases[%d] = %f\n", i, biases[i]);
    }
  }

  free(neuron_Values);
  free(errors);
  free(gradients);
  free(bias_Gradients);
  free(weights);
  free(biases);
}

int main() {
  test_forward_and_backward_pass();
  return 0;
}
