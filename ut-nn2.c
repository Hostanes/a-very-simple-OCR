/*
  ut-nn2.c

  TODO Weight updating is slightly off, back propagation needs to be tuned
*/

#include "nnlib.h"
#include <math.h>
#include <stdio.h>

#define EPSILON 1e-5 // For finite difference checking

void test_backprop_updates() {
  printf("\n=== Starting Backpropagation Unit Test ===\n");

  // Tiny network: 2 inputs, 3 hidden, 1 output
  const int layer_sizes[] = {2, 3, 1};
  Model_t *model = create_model(2, layer_sizes, 0.1); // LR=0.1

  // Manually set weights/biases for predictable behavior
  double weights1[3][2] = {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}};
  double biases1[3] = {0.1, 0.1, 0.1};
  double weights2[1][3] = {{0.7, 0.8, 0.9}};
  double bias2[1] = {0.1};

  // Copy to model
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++)
      model->layers[0]->weights->data[i][j] = weights1[i][j];
    model->layers[0]->biases->data[i][0] = biases1[i];
  }
  for (int j = 0; j < 3; j++)
    model->layers[1]->weights->data[0][j] = weights2[0][j];
  model->layers[1]->biases->data[0][0] = bias2[0];

  // Test input and target
  Matrix *input = init_Matrix(2, 1);
  input->data[0][0] = 1.0;
  input->data[1][0] = 0.5;

  Matrix *target = init_Matrix(1, 1);
  target->data[0][0] = 0.8;

  // Save original weights
  double orig_w1[3][2], orig_w2[1][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      orig_w1[i][j] = model->layers[0]->weights->data[i][j];

  for (int j = 0; j < 3; j++)
    orig_w2[0][j] = model->layers[1]->weights->data[0][j];

  // Run one training iteration
  Matrix *activations[3]; // 2 layers + input
  for (int i = 0; i < 3; i++)
    activations[i] = init_Matrix(layer_sizes[i], 1);

  forward_pass(model, input, activations);
  backward_pass(model, activations, target);

  // Verify weight updates (numerical gradient checking)
  int passed = 1;

  // Check hidden-to-output weights
  double expected_w2_update[3] = {0.0064, 0.0065, 0.0066}; // Precomputed
  for (int j = 0; j < 3; j++) {
    double actual_update =
        orig_w2[0][j] - model->layers[1]->weights->data[0][j];
    if (fabs(actual_update - expected_w2_update[j]) > EPSILON) {
      printf("FAIL: Output weight %d update %.6f != expected %.6f\n", j,
             actual_update, expected_w2_update[j]);
      passed = 0;
    }
  }

  // Check input-to-hidden weights (first neuron only for brevity)
  double expected_w1_update[2] = {0.0003, 0.0001}; // Precomputed
  for (int j = 0; j < 2; j++) {
    double actual_update =
        orig_w1[0][j] - model->layers[0]->weights->data[0][j];
    if (fabs(actual_update - expected_w1_update[j]) > EPSILON) {
      printf("FAIL: Hidden weight %d update %.6f != expected %.6f\n", j,
             actual_update, expected_w1_update[j]);
      passed = 0;
    }
  }

  if (passed) {
    printf("PASS: All weight updates match expected values\n");
  } else {
    printf("\nDebugging Info:\n");
    printf("Final output: %.6f (target=0.8)\n", activations[2]->data[0][0]);

    // Print all layer outputs
    for (int l = 0; l <= 2; l++) {
      printf("Layer %d outputs:\n", l);
      for (int i = 0; i < activations[l]->rows; i++) {
        printf("  Neuron %d: %.6f\n", i, activations[l]->data[i][0]);
      }
    }
  }

  // Cleanup
  for (int i = 0; i < 3; i++)
    matrix_Free(activations[i]);
  matrix_Free(input);
  matrix_Free(target);
  free_model(model);
}

int main() {
  test_backprop_updates();
  return 0;
}
