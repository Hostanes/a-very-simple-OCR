#include "lib/matrix-math.h"
// #include "lib-omp/matrix-math.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Network architecture
#define NUM_LAYERS 8
const int layer_sizes[NUM_LAYERS + 1] = {8192, 4096, 2048, 1024, 512,
                                         256,  128,  64,   16};

// Training parameters
#define LEARNING_RATE 0.01
#define EPOCHS 10

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

// Initialize biases with small random values
void random_bias_init(Matrix *biases) {
  for (int i = 0; i < biases->rows; i++) {
    biases->data[i][0] = (double)rand() / RAND_MAX * 0.01;
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

// ReLU derivative (for backpropagation)
void ReLU_derivative(Matrix *mat) {
  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->columns; j++) {
      mat->data[i][j] = mat->data[i][j] > 0 ? 1.0 : 0.0;
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

// Forward propagation through one layer
void forward_Prop_Layer(Matrix *input, Matrix *output, Matrix *weights,
                        Matrix *bias, void (*activationFunc)(Matrix *mat)) {
  Matrix *temp = dot_Mat(weights, input);

  for (int i = 0; i < output->rows; i++) {
    output->data[i][0] = temp->data[i][0] + bias->data[i][0];
  }

  activationFunc(output);
  matrix_Free(temp);
}

// Backpropagation
void backpropagate(Matrix *activations[], Matrix *weights[], Matrix *biases[],
                   Matrix *target) {
  // Calculate output error (cross-entropy loss derivative with softmax)
  Matrix *output_error = init_Matrix(layer_sizes[NUM_LAYERS], 1);
  for (int i = 0; i < layer_sizes[NUM_LAYERS]; i++) {
    output_error->data[i][0] =
        activations[NUM_LAYERS]->data[i][0] - target->data[i][0];
  }

  // Backpropagate through layers
  Matrix *delta = output_error;
  for (int l = NUM_LAYERS - 1; l >= 0; l--) {
    // Calculate gradient for weights
    Matrix *activation_transpose = init_Matrix(1, layer_sizes[l]);
    for (int i = 0; i < layer_sizes[l]; i++) {
      activation_transpose->data[0][i] = activations[l]->data[i][0];
    }

    Matrix *weight_gradient = dot_Mat(delta, activation_transpose);

    // Update weights
    for (int i = 0; i < weights[l]->rows; i++) {
      for (int j = 0; j < weights[l]->columns; j++) {
        weights[l]->data[i][j] -= LEARNING_RATE * weight_gradient->data[i][j];
      }
    }

    // Update biases
    for (int i = 0; i < biases[l]->rows; i++) {
      biases[l]->data[i][0] -= LEARNING_RATE * delta->data[i][0];
    }

    // Calculate error for previous layer (except for input layer)
    if (l > 0) {
      Matrix *weights_transpose =
          init_Matrix(weights[l]->columns, weights[l]->rows);
      for (int i = 0; i < weights[l]->rows; i++) {
        for (int j = 0; j < weights[l]->columns; j++) {
          weights_transpose->data[j][i] = weights[l]->data[i][j];
        }
      }

      Matrix *prev_error = dot_Mat(weights_transpose, delta);

      // Apply ReLU derivative
      Matrix *relu_deriv = init_Matrix(layer_sizes[l], 1);
      for (int i = 0; i < layer_sizes[l]; i++) {
        relu_deriv->data[i][0] = activations[l]->data[i][0] > 0 ? 1.0 : 0.0;
      }

      // Element-wise multiplication
      for (int i = 0; i < layer_sizes[l]; i++) {
        prev_error->data[i][0] *= relu_deriv->data[i][0];
      }

      // Prepare for next iteration
      matrix_Free(delta);
      delta = prev_error;
      matrix_Free(relu_deriv);
      matrix_Free(weights_transpose);
    }

    matrix_Free(activation_transpose);
    matrix_Free(weight_gradient);
  }

  matrix_Free(delta);
}

int main() {
  srand(time(NULL));

  // Initialize network
  Matrix *weights[NUM_LAYERS];
  Matrix *biases[NUM_LAYERS];
  Matrix *activations[NUM_LAYERS + 1];

  for (int i = 0; i < NUM_LAYERS; i++) {
    weights[i] = init_Matrix(layer_sizes[i + 1], layer_sizes[i]);
    xavier_init(weights[i]);
    biases[i] = init_Matrix(layer_sizes[i + 1], 1);
    random_bias_init(biases[i]);
  }

  for (int i = 0; i <= NUM_LAYERS; i++) {
    activations[i] = init_Matrix(layer_sizes[i], 1);
  }

  double start_time = omp_get_wtime();

  // Training loop
  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    printf("Epoch %d\n", epoch + 1);

    // Generate random input and target (for demonstration)
    for (int i = 0; i < layer_sizes[0]; i++) {
      activations[0]->data[i][0] = (double)rand() / RAND_MAX;
    }

    Matrix *target = init_Matrix(layer_sizes[NUM_LAYERS], 1);
    target->data[rand() % layer_sizes[NUM_LAYERS]][0] = 1.0; // One-hot target

    // Forward pass
    for (int i = 0; i < NUM_LAYERS - 1; i++) {
      forward_Prop_Layer(activations[i], activations[i + 1], weights[i],
                         biases[i], ReLU);
    }
    forward_Prop_Layer(activations[NUM_LAYERS - 1], activations[NUM_LAYERS],
                       weights[NUM_LAYERS - 1], biases[NUM_LAYERS - 1],
                       softmax);

    // Backward pass
    backpropagate(activations, weights, biases, target);

    // Print loss (cross-entropy)
    double loss = 0.0;
    for (int i = 0; i < layer_sizes[NUM_LAYERS]; i++) {
      if (target->data[i][0] == 1.0) {
        loss = -log(activations[NUM_LAYERS]->data[i][0]);
        break;
      }
    }
    printf("Loss: %.4f\n", loss);

    matrix_Free(target);
  }

  double end_time = omp_get_wtime();
  printf("time taken: %f\n", end_time - start_time);

  // Cleanup
  for (int i = 0; i < NUM_LAYERS; i++) {
    matrix_Free(weights[i]);
    matrix_Free(biases[i]);
  }
  for (int i = 0; i <= NUM_LAYERS; i++) {
    matrix_Free(activations[i]);
  }

  return 0;
}
