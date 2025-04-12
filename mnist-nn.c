/*
  run the neural network on the MNIST library, too small to notice differences
  with parallelization
*/

#define USE_MNIST_LOADER
#define MNIST_BINARY
#include "mnist-dataloader/mnist-dataloader.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "lib/matrix-math.h"
#include "nnlib.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Network architecture
#define NUM_LAYERS 4
const int layer_sizes[NUM_LAYERS + 1] = {784, 256, 128, 64, 10};

// Training parameters
#define LEARNING_RATE 0.0001
#define EPOCHS 10

int main(int argc, char **argv) {
  // Load MNIST data (unchanged)
  mnist_data *train_data;
  unsigned int train_count;
  if (mnist_load("dataset/train-images.idx3-ubyte",
                 "dataset/train-labels.idx1-ubyte", &train_data,
                 &train_count)) {
    printf("Error loading MNIST data\n");
    return 1;
  }
  printf("Loaded %d training images\n", train_count);

  // Create model
  Model_t *model = create_model(NUM_LAYERS, layer_sizes, LEARNING_RATE);

  /*
    Prepare batch containers
    Train function expects inputs and targets in seperate arrays
    so we must convert mnist data to these 2 arrays
  */
  Matrix *inputs[train_count];
  Matrix *targets[train_count];

  for (int i = 0; i < train_count; i++) {
    // Create input matrix (flatten 28x28 to 784x1)
    inputs[i] = init_Matrix(784, 1);
    for (int p = 0; p < 784; p++) {
      inputs[i]->data[p][0] = train_data[i].data[p] / 255.0; // Normalize [0-1]
    }

    // Create target
    targets[i] = init_Matrix(10, 1);
    targets[i]->data[train_data[i].label][0] = 1.0;
  }

  train_model_batch(model, inputs, targets, train_count/5, EPOCHS, LEARNING_RATE);

  for (int i = 0; i < train_count; i++) {
    matrix_Free(inputs[i]);
    matrix_Free(targets[i]);
  }
  free_model(model);
  free(train_data);

  return 0;
}
