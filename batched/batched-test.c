
#include "lib/nnlib.h"
#include <stdio.h>
#include <stdlib.h>

int main() {

  srand(78);
  
  // Set up a small neural network: 2 inputs, 3 hidden, 2 outputs
  int layer_sizes[] = {2, 3, 2};
  int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);

  // Use ReLU for hidden layer and softmax for output
  ActivationFunc activations[] = {relu, softmax_placeholder};
  ActivationDerivative derivatives[] = {relu_derivative, softmax_placeholder};

  // Create network with batch size of 3
  NeuralNetwork_t *net = create_network(layer_sizes, num_layers, activations,
                                        derivatives, 0.01f, 0.9f, 3);

  // Create 6 training samples (2 batches of 3)
  float inputs[6][2] = {
      {0.1f, 0.2f}, // Sample 1
      {0.3f, 0.4f}, // Sample 2
      {0.5f, 0.6f}, // Sample 3
      {0.7f, 0.8f}, // Sample 4
      {0.9f, 1.0f}, // Sample 5
      {1.1f, 1.2f}  // Sample 6
  };

  // Corresponding targets (one-hot encoded)
  float targets[6][2] = {
      {1.0f, 0.0f}, // Class 0
      {0.0f, 1.0f}, // Class 1
      {1.0f, 0.0f}, // Class 0
      {0.0f, 1.0f}, // Class 1
      {1.0f, 0.0f}, // Class 0
      {0.0f, 1.0f}  // Class 1
  };

  printf("Training the network with 2 batches of 3 samples each...\n");

  // Train in 2 batches
  for (int epoch = 0; epoch < 10; epoch++) {
    printf("\nEpoch %d\n", epoch + 1);

    // First batch (samples 0-2)
    train_batch(net, (float *)inputs, (float *)targets, 3);

    // Second batch (samples 3-5)
    train_batch(net, (float *)inputs + 6, (float *)targets + 6, 3);

    // Calculate and print loss for all samples
    float *outputs = forward_pass_batch(net, (float *)inputs, 6);
    float loss = calculate_batch_loss(net, outputs, (float *)targets, 6);
    printf("Loss: %.4f\n", loss);

    // Print predictions for first sample in each batch
    int *preds1 = predict_batch(net, (float *)inputs, 3);
    printf("Batch 1 predictions: %d, %d, %d\n", preds1[0], preds1[1],
           preds1[2]);
    free(preds1);

    int *preds2 = predict_batch(net, (float *)inputs + 6, 3);
    printf("Batch 2 predictions: %d, %d, %d\n", preds2[0], preds2[1],
           preds2[2]);
    free(preds2);
  }

  // Test the network with some new data
  float test_input[3][2] = {
      {0.2f, 0.3f}, // Should predict class 0
      {0.4f, 0.9f}, // Should predict class 1
      {0.6f, 0.7f}  // Should predict class 0
  };

  printf("\nTesting the network...\n");
  int *test_preds = predict_batch(net, (float *)test_input, 3);
  printf("Test predictions: %d, %d, %d\n", test_preds[0], test_preds[1],
         test_preds[2]);

  // Print final outputs for test samples
  float *test_outputs = forward_pass_batch(net, (float *)test_input, 3);
  printf("Test outputs:\n");
  for (int i = 0; i < 3; i++) {
    printf("Sample %d: [%.4f, %.4f]\n", i + 1, test_outputs[i * 2],
           test_outputs[i * 2 + 1]);
  }

  free(test_preds);
  free_network(net);

  return 0;
}
