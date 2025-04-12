/*
  random-nn-test.c

  Test out the effect of parallelization on the neural network

  data is completely randomized, therefore accuracy is spread out equally
  across the 10 output classes (~10%)
*/

#include "lib/matrix-math.h"
#include "nnlib.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "lib-omp/config.h"

// Larger network architecture for benchmarking
#define NUM_LAYERS 5
const int layer_sizes[NUM_LAYERS + 1] = {1024, 512, 256,
                                         128,  64,  10}; // 1K input, 10 output

// Benchmark parameters
#define LEARNING_RATE 0.0001
#define EPOCHS 5
#define BATCH_SIZE 10000 // Larger batch for better timing

int main() {

  omp_set_num_threads(16);

  srand(time(NULL));
  printf("=== Neural Network Speed Benchmark ===\n");
  printf("Network Architecture: ");
  for (int i = 0; i <= NUM_LAYERS; i++)
    printf("%d ", layer_sizes[i]);
  printf("\n");

  // Create model
  Model_t *model = create_model(NUM_LAYERS, layer_sizes, LEARNING_RATE);

  // Generate random training data
  Matrix *inputs[BATCH_SIZE];
  Matrix *targets[BATCH_SIZE];

  printf("Generating %d random samples...\n", BATCH_SIZE);
#ifdef USE_OPENMP
  printf("using openmp\n");
#pragma omp parallel for
#endif
  for (int i = 0; i < BATCH_SIZE; i++) {
    inputs[i] = init_Matrix(layer_sizes[0], 1);
    targets[i] = init_Matrix(layer_sizes[NUM_LAYERS], 1);

    // Random input (0-1)
    for (int j = 0; j < layer_sizes[0]; j++) {
      inputs[i]->data[j][0] = (double)rand() / RAND_MAX;
    }

    // Random one-hot target
    int random_class = rand() % layer_sizes[NUM_LAYERS];
    targets[i]->data[random_class][0] = 1.0;
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

  // Cleanup
  for (int i = 0; i < BATCH_SIZE; i++) {
    matrix_Free(inputs[i]);
    matrix_Free(targets[i]);
  }
  free_model(model);

  return 0;
}
