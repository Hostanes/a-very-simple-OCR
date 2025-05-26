
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
  int input_size = 4;
  int layer_sizes[] = {3, 2, 2}; // 3-layer network: 4->3->2->2
  int num_layers = 3;

  // Activation functions for each layer
  ActivationFunc activations[] = {relu, relu, softmax_placeholder};
  ActivationDerivative derivs[] = {relu_derivative, relu_derivative, NULL};

  // Create network
  NeuralNetwork_t *net = create_network(input_size, layer_sizes, num_layers,
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
  printf("Layer sizes: %d -> %d -> %d -> %d\n", input_size, layer_sizes[0],
         layer_sizes[1], layer_sizes[2]);
  printf("Learning rate: %.3f, Momentum: %.3f\n\n", net->learning_Rate,
         net->momentum);

  // Test sample 1
  printf("=== Testing Sample 1 ===\n");
  print_array(sample1_input, input_size, "Input");

  forward_Pass(net, sample1_input);
  print_array(net->A_values + net->A_Offsets[num_layers - 1],
              layer_sizes[num_layers - 1], "Output before training");

  backward_Pass(net, sample1_input, sample1_target);
  forward_Pass(net, sample1_input); // << Run forward pass after weight update
  print_array(net->A_values + net->A_Offsets[num_layers - 1],
              layer_sizes[num_layers - 1], "Output after 1 training step");

  // Test sample 2
  printf("\n=== Testing Sample 2 ===\n");
  print_array(sample2_input, input_size, "Input");

  forward_Pass(net, sample2_input);
  print_array(net->A_values + net->A_Offsets[num_layers - 1],
              layer_sizes[num_layers - 1], "Output before training");

  backward_Pass(net, sample2_input, sample2_target);
  forward_Pass(net, sample2_input); // << Run forward pass after weight update
  print_array(net->A_values + net->A_Offsets[num_layers - 1],
              layer_sizes[num_layers - 1], "Output after 1 training step");

  // Test predictions after training
  printf("\n=== Prediction Test ===\n");
  forward_Pass(net, sample1_input);
  float *output = net->A_values + net->A_Offsets[num_layers - 1];
  printf("Sample 1 prediction: [%.4f, %.4f] -> Predicted class: %d\n",
         output[0], output[1], (output[0] > output[1]) ? 0 : 1);

  forward_Pass(net, sample2_input);
  output = net->A_values + net->A_Offsets[num_layers - 1];
  printf("Sample 2 prediction: [%.4f, %.4f] -> Predicted class: %d\n",
         output[0], output[1], (output[0] > output[1]) ? 0 : 1);

  // Clean up
  free_network(net);
  return 0;
}
