#include "lib/nnlib.h"
#include <stdio.h>
#include <time.h>

void print_array(float *arr, int size, const char *name) {
  printf("%s: [", name);
  for (int i = 0; i < size; i++) {
    printf("%.4f", arr[i]);
    if (i < size - 1)
      printf(", ");
  }
  printf("]\n");
}

int main() {
  // Set random seed for reproducible results
  srand(42);

  // Network configuration
  int layer_sizes[] = {4, 3, 2, 2}; // Input layer included: 4->3->2->2
  int num_layers = 3; // Number of hidden + output layers (excluding input)

  // Activation functions for each layer (excluding input layer)
  ActivationFunc activations[] = {relu, relu, softmax_placeholder};
  ActivationDerivative derivs[] = {relu_derivative, relu_derivative, NULL};

  // Create network
  NeuralNetwork_t *net = create_network(
      layer_sizes, num_layers + 1, // +1 to include input layer size
      activations, derivs, 0.01f, 0.9f);
  if (!net) {
    printf("Failed to create network!\n");
    return 1;
  }

  // Test samples (input and expected output)
  float sample1_input[] = {0.5f, 0.3f, 0.2f, 0.8f};
  float sample1_target[] = {1.0f, 0.0f}; // Class 0

  float sample2_input[] = {0.1f, 0.9f, 0.4f, 0.6f};
  float sample2_target[] = {0.0f, 1.0f}; // Class 1

  printf("=== Initial Network ===\n");
  printf("Layer sizes: %d -> %d -> %d -> %d\n", layer_sizes[0], layer_sizes[1],
         layer_sizes[2], layer_sizes[3]);
  printf("Learning rate: %.3f, Momentum: %.3f\n\n", net->learning_rate,
         net->momentum);

  // Test sample 1
  printf("=== Testing Sample 1 ===\n");
  print_array(sample1_input, layer_sizes[0], "Input");

  // Forward pass and get output
  float *output1 = forward_pass(net, sample1_input);
  print_array(output1, layer_sizes[3], "Output before training");

  // Backward pass (training step)
  backward_pass(net, sample1_input, sample1_target);

  // Forward pass again to see the change
  output1 = forward_pass(net, sample1_input);
  print_array(output1, layer_sizes[3], "Output after 1 training step");

  // Test sample 2
  printf("\n=== Testing Sample 2 ===\n");
  print_array(sample2_input, layer_sizes[0], "Input");

  // Forward pass and get output
  float *output2 = forward_pass(net, sample2_input);
  print_array(output2, layer_sizes[3], "Output before training");

  // Backward pass (training step)
  backward_pass(net, sample2_input, sample2_target);

  // Forward pass again to see the change
  output2 = forward_pass(net, sample2_input);
  print_array(output2, layer_sizes[3], "Output after 1 training step");

  // Test predictions after training
  printf("\n=== Prediction Test ===\n");

  // Test sample 1 prediction
  output1 = forward_pass(net, sample1_input);
  printf("Sample 1 prediction: [%.4f, %.4f] -> Predicted class: %d\n",
         output1[0], output1[1], (output1[0] > output1[1]) ? 0 : 1);

  // Test sample 2 prediction
  output2 = forward_pass(net, sample2_input);
  printf("Sample 2 prediction: [%.4f, %.4f] -> Predicted class: %d\n",
         output2[0], output2[1], (output2[0] > output2[1]) ? 0 : 1);

  // You could also use the predict function for cleaner prediction
  printf("\nUsing predict function:\n");
  int pred1 = predict(net, sample1_input);
  int pred2 = predict(net, sample2_input);
  printf("Sample 1 predicted class: %d\n", pred1);
  printf("Sample 2 predicted class: %d\n", pred2);

  // Clean up
  free_network(net);

  return 0;
}
