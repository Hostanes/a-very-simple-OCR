/*
  left-right-nn.c

  tests the accuracy of the DNN using a very simple 2 class classification
  some of the matricies have piexls on the left side, others have pixels on the
  right side

  results in immediate 100% accuracy by second epoch, used as an occam's razor
  to confirm weight update works
*/
#include "lib/matrix-math.h"
#include "nnlib.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "lib-omp/config.h"

// Larger network architecture
#define NUM_LAYERS 5
const int layer_sizes[NUM_LAYERS + 1] = {
    1024, 512, 256, 128, 64, 2}; // 32x32 input, 2 output classes

// Enhanced benchmark parameters
#define LEARNING_RATE 0.001
#define EPOCHS 10
#define BATCH_SIZE 5000     // Larger sample size
#define IMG_SIZE 32         // 32x32 images
#define PATTERN_THICKNESS 8 // Width of the pattern border

int main() {
  omp_set_num_threads(16); // Set to your core count
  srand(time(NULL));

  printf("=== Large-Scale Pattern Classification Benchmark ===\n");
  printf("Network Architecture: ");
  for (int i = 0; i <= NUM_LAYERS; i++)
    printf("%d ", layer_sizes[i]);
  printf("\nInput: %dx%d images\n", IMG_SIZE, IMG_SIZE);
  printf("Batch size: %d samples\n", BATCH_SIZE);

  // Create model
  Model_t *model = create_model(NUM_LAYERS, layer_sizes, LEARNING_RATE);

  // Generate structured training data
  Matrix *inputs[BATCH_SIZE];
  Matrix *targets[BATCH_SIZE];

  printf("Generating %d pattern samples...\n", BATCH_SIZE);
#pragma omp parallel for
  for (int i = 0; i < BATCH_SIZE; i++) {
    inputs[i] = init_Matrix(IMG_SIZE * IMG_SIZE, 1);
    targets[i] = init_Matrix(2, 1);

    // Create clear left/right pattern with gradient
    int pattern_type = i % 2; // Alternating classes

    for (int y = 0; y < IMG_SIZE; y++) {
      for (int x = 0; x < IMG_SIZE; x++) {
        // Left pattern (class 0)
        if (pattern_type == 0) {
          float value = (x < IMG_SIZE / 2 - PATTERN_THICKNESS) ? 0.9
                        : (x < IMG_SIZE / 2)
                            ? 0.5 + 0.4 * (IMG_SIZE / 2 - x) / PATTERN_THICKNESS
                            : 0.1;
          inputs[i]->data[y * IMG_SIZE + x][0] = value;
        }
        // Right pattern (class 1)
        else {
          float value = (x > IMG_SIZE / 2 + PATTERN_THICKNESS) ? 0.9
                        : (x > IMG_SIZE / 2)
                            ? 0.5 + 0.4 * (x - IMG_SIZE / 2) / PATTERN_THICKNESS
                            : 0.1;
          inputs[i]->data[y * IMG_SIZE + x][0] = value;
        }
      }
    }

    // Set one-hot target
    targets[i]->data[pattern_type][0] = 1.0;
  }

  // Benchmark training
  printf("Starting training benchmark (%d epochs)...\n", EPOCHS);
  double start_time = omp_get_wtime();

  train_model_batch(model, inputs, targets, BATCH_SIZE, EPOCHS, LEARNING_RATE);

  double total_time = omp_get_wtime() - start_time;
  printf("\n=== Benchmark Results ===\n");
  printf("Total time: %.2f seconds\n", total_time);
  printf("Time per epoch: %.2f seconds\n", total_time / EPOCHS);
  printf("Samples per second: %.0f\n", (BATCH_SIZE * EPOCHS) / total_time);

  // Test prediction
  printf("\nTesting predictions:\n");
  Matrix *test = init_Matrix(IMG_SIZE * IMG_SIZE, 1);

  // Create left pattern test case with gradient
  for (int y = 0; y < IMG_SIZE; y++) {
    for (int x = 0; x < IMG_SIZE; x++) {
      float value = (x < IMG_SIZE / 2 - PATTERN_THICKNESS) ? 1.0
                    : (x < IMG_SIZE / 2)
                        ? 0.5 + 0.5 * (IMG_SIZE / 2 - x) / PATTERN_THICKNESS
                        : 0.0;
      test->data[y * IMG_SIZE + x][0] = value;
    }
  }

  Matrix *activations[NUM_LAYERS + 1];
  for (int i = 0; i <= NUM_LAYERS; i++) {
    activations[i] = init_Matrix(layer_sizes[i], 1);
  }

  forward_pass(model, test, activations);
  printf("Left pattern prediction: [%.3f, %.3f] (should be [~1, ~0])\n",
         activations[NUM_LAYERS]->data[0][0],
         activations[NUM_LAYERS]->data[1][0]);

  // Cleanup
  for (int i = 0; i < BATCH_SIZE; i++) {
    matrix_Free(inputs[i]);
    matrix_Free(targets[i]);
  }
  for (int i = 0; i <= NUM_LAYERS; i++) {
    matrix_Free(activations[i]);
  }
  matrix_Free(test);
  free_model(model);

  return 0;
}
