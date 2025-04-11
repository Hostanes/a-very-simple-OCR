#include "lib/matrix-math.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Network architecture
#define NUM_LAYERS 8
const int layer_sizes[NUM_LAYERS + 1] = {8192, 4096, 2048, 1024,
                                         512,  256,  128, 64,  28};

// Xavier/Glorot initialization scaling factor
double xavier_scale(int input_size) { return sqrt(2.0 / input_size); }

// Initialize weights with Xavier/Glorot initialization
void xavier_init(Matrix *weights) {
  double scale = xavier_scale(weights->columns);
  for (int i = 0; i < weights->rows; i++) {
    for (int j = 0; j < weights->columns; j++) {
      weights->data[i][j] = (2.0 * rand() / RAND_MAX - 1.0) * scale;
    }
  }
}

// Initialize biases with small random values instead of zeros
void random_bias_init(Matrix *biases) {
  for (int i = 0; i < biases->rows; i++) {
    biases->data[i][0] =
        (double)rand() / RAND_MAX * 0.01; // Small random values
  }
}

// Activation functions
void ReLU(Matrix *mat) {
  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->columns; j++) {
      mat->data[i][j] = fmax(0.0, mat->data[i][j]);
    }
  }
}

// Numerically stable softmax
void softmax(Matrix *mat) {
  double max_val = mat->data[0][0];
  for (int i = 1; i < mat->rows; i++) {
    if (mat->data[i][0] > max_val)
      max_val = mat->data[i][0];
  }

  double sum = 0.0;
  for (int i = 0; i < mat->rows; i++) {
    mat->data[i][0] = exp(mat->data[i][0] - max_val);
    sum += mat->data[i][0];
  }
  for (int i = 0; i < mat->rows; i++) {
    mat->data[i][0] /= sum;
  }
}

// Forward propagation through one layer (with bias)
void forward_Prop_Layer(Matrix *input, Matrix *output, Matrix *weights,
                        Matrix *bias, void (*activationFunc)(Matrix *mat)) {
  Matrix *temp = dot_Mat(weights, input);

  // Add bias to each element
  for (int i = 0; i < output->rows; i++) {
    output->data[i][0] = temp->data[i][0] + bias->data[i][0];
  }

  // Apply activation function
  activationFunc(output);

  matrix_Free(temp);
}

int main() {
  // Initialize random seed
  srand(time(NULL));

  // Create arrays to hold weights and biases
  Matrix *weights[NUM_LAYERS];
  Matrix *biases[NUM_LAYERS];

  // Initialize weights and biases
  for (int i = 0; i < NUM_LAYERS; i++) {
    weights[i] = init_Matrix(layer_sizes[i + 1], layer_sizes[i]);
    xavier_init(weights[i]);

    biases[i] = init_Matrix(layer_sizes[i + 1], 1);
    random_bias_init(biases[i]);
  }

  // Initialize activations
  Matrix *activations[NUM_LAYERS + 1]; // +1 for input layer
  for (int i = 0; i <= NUM_LAYERS; i++) {
    activations[i] = init_Matrix(layer_sizes[i], 1);
  }

  // Randomize input (normalized between 0 and 1)
  for (int i = 0; i < layer_sizes[0]; i++) {
    activations[0]->data[i][0] = (double)rand() / RAND_MAX;
  }

  // Forward propagation through all layers except last
  for (int i = 0; i < NUM_LAYERS - 1; i++) {
    forward_Prop_Layer(activations[i], activations[i + 1], weights[i],
                       biases[i], ReLU);
  }

  // Final layer with softmax
  forward_Prop_Layer(activations[NUM_LAYERS - 1], activations[NUM_LAYERS],
                     weights[NUM_LAYERS - 1], biases[NUM_LAYERS - 1], softmax);

  // Print output
  printf("Final output probabilities:\n");
  print_Matrix(activations[NUM_LAYERS]);

  // Free memory
  for (int i = 0; i < NUM_LAYERS; i++) {
    matrix_Free(weights[i]);
    matrix_Free(biases[i]);
  }

  for (int i = 0; i <= NUM_LAYERS; i++) {
    matrix_Free(activations[i]);
  }

  return 0;
}
