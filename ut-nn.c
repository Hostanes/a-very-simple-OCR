
#include "nnlib.h"
#include <assert.h>
#include <stdio.h>
#include <time.h>

void test_neural_network() {
  printf("=========================================\n");
  printf("Starting neural network unit tests...\n");

  /*
    Test Architechture:
    input 64
    hidden 1 32
    hidden 2 16
    output 10
  */
  const int input_size = 8 * 8;
  const int hidden_size1 = 32;
  const int hidden_size2 = 16;
  const int output_size = 10;
  const double learning_rate = 0.01;
  const int epochs = 5;

  // Layer sizes array
  int layer_sizes[] = {input_size, hidden_size1, hidden_size2, output_size};
  int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]) - 1;

  // Create model
  Model_t *model = create_model(num_layers, layer_sizes, learning_rate);
  printf("Model created\n");

  // Test model structure
  assert(model != NULL);
  assert(model->num_layers == 3);
  assert(model->learning_rate == learning_rate);

  // Test layer configurations
  assert(model->layers[0]->input_size == input_size);
  assert(model->layers[0]->output_size == hidden_size1);
  assert(model->layers[1]->input_size == hidden_size1);
  assert(model->layers[1]->output_size == hidden_size2);
  assert(model->layers[2]->input_size == hidden_size2);
  assert(model->layers[2]->output_size == output_size);

  // Test activation functions
  assert(model->layers[0]->activation == relu);
  assert(model->layers[0]->activation_derivative == relu_derivative);
  assert(model->layers[1]->activation == relu);
  assert(model->layers[1]->activation_derivative == relu_derivative);
  assert(model->layers[2]->activation == softmax);
  assert(model->layers[2]->activation_derivative == NULL);

  printf("Model structure tests passed\n");

  // Create test input (8x8 matrix flattened to 64x1)
  Matrix *input = init_Matrix(input_size, 1);
  for (int i = 0; i < input_size; i++) {
    input->data[i][0] =
        (double)rand() / RAND_MAX; // Random values between 0 and 1
  }

  // Initialize activations for forward pass
  Matrix *activations[num_layers + 1];
  for (int i = 0; i <= num_layers; i++) {
    int size = (i == 0) ? input_size : model->layers[i - 1]->output_size;
    activations[i] = init_Matrix(size, 1);
  }

  // Test forward pass
  forward_pass(model, input, activations);
  printf("Forward pass completed successfully.\n");

  // Verify activation shapes
  assert(activations[0]->rows == input_size);
  assert(activations[1]->rows == hidden_size1);
  assert(activations[2]->rows == hidden_size2);
  assert(activations[3]->rows == output_size);

  // Verify ReLU activation (no negative values in hidden layers)
  for (int i = 0; i < hidden_size1; i++) {
    assert(activations[1]->data[i][0] >= 0);
  }
  for (int i = 0; i < hidden_size2; i++) {
    assert(activations[2]->data[i][0] >= 0);
  }

  // Verify softmax output (sums to ~1)
  double sum = 0.0;
  for (int i = 0; i < output_size; i++) {
    assert(activations[3]->data[i][0] >= 0);
    assert(activations[3]->data[i][0] <= 1);
    sum += activations[3]->data[i][0];
  }
  assert(fabs(sum - 1.0) < 1e-6);

  printf("Forward pass tests passed.\n");

  // Create test target (one-hot encoded)
  Matrix *target = init_Matrix(output_size, 1);
  int target_class = rand() % output_size;
  target->data[target_class][0] = 1.0;

  // Test backward pass
  backward_pass(model, activations, target);
  printf("Backward pass completed successfully.\n");

  // Training test
  printf("Starting training test (%d epochs)...\n", epochs);
  train_model(model, epochs);
  printf("Training test completed.\n");

  // Cleanup
  for (int i = 0; i <= num_layers; i++) {
    matrix_Free(activations[i]);
  }
  matrix_Free(input);
  matrix_Free(target);
  free_model(model);

  printf("=========================================\n");

  printf("All neural network tests passed!\n");
}

int main() {
  srand(time(0));

  test_neural_network();
  return 0;
}
