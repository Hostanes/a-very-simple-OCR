
#include "../lib/nnlib-ocl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Helper function to compare float arrays
int compareFloatArrays(const float *arr1, const float *arr2, size_t size,
                       float epsilon) {
  for (size_t i = 0; i < size; i++) {
    if (fabs(arr1[i] - arr2[i]) > epsilon) {
      printf("Difference at index %zu: expected %f, got %f\n", i, arr1[i],
             arr2[i]);
      return 0; // Arrays are not equal
    }
  }
  return 1; // Arrays are equal
}

int main() {
  NeuralNetwork_t network;
  int num_layers = 4;
  size_t layer_sizes[] = {2, 3, 4, 2}; // Input (2), Hidden (3, 4), Output (2)
  float learning_rate = 0.1f;
  cl_int error;

  // Initialize the neural network
  if (initNeuralNetwork(&network, num_layers, layer_sizes) != 0) {
    fprintf(stderr, "Failed to initialize neural network\n");
    return 1;
  }

  // 1. Set layer data (weights and biases)
  // Layer 1: 2 input, 3 output
  float layer1_weights_data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
  float layer1_biases_data[] = {0.1f, 0.2f, 0.3f};

  // Layer 2: 3 input, 4 output
  float layer2_weights_data[] = {0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f,
                                 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f};
  float layer2_biases_data[] = {0.4f, 0.5f, 0.6f, 0.7f};

  // Layer 3: 4 input, 2 output
  float layer3_weights_data[] = {1.9f, 2.0f, 2.1f, 2.2f,
                                 2.3f, 2.4f, 2.5f, 2.6f};
  float layer3_biases_data[] = {0.8f, 0.9f};

  // Copy data to OpenCL buffers
  error = clEnqueueWriteBuffer(network.queue, network.layers[0].weights,
                               CL_TRUE, 0, sizeof(layer1_weights_data),
                               layer1_weights_data, 0, NULL, NULL);
  if (error != CL_SUCCESS) {
    destroyNeuralNetwork(&network);
    return 1;
  }
  error = clEnqueueWriteBuffer(network.queue, network.layers[0].biases, CL_TRUE,
                               0, sizeof(layer1_biases_data),
                               layer1_biases_data, 0, NULL, NULL);
  if (error != CL_SUCCESS) {
    destroyNeuralNetwork(&network);
    return 1;
  }

  error = clEnqueueWriteBuffer(network.queue, network.layers[1].weights,
                               CL_TRUE, 0, sizeof(layer2_weights_data),
                               layer2_weights_data, 0, NULL, NULL);
  if (error != CL_SUCCESS) {
    destroyNeuralNetwork(&network);
    return 1;
  }
  error = clEnqueueWriteBuffer(network.queue, network.layers[1].biases, CL_TRUE,
                               0, sizeof(layer2_biases_data),
                               layer2_biases_data, 0, NULL, NULL);
  if (error != CL_SUCCESS) {
    destroyNeuralNetwork(&network);
    return 1;
  }

  error = clEnqueueWriteBuffer(network.queue, network.layers[2].weights,
                               CL_TRUE, 0, sizeof(layer3_weights_data),
                               layer3_weights_data, 0, NULL, NULL);
  if (error != CL_SUCCESS) {
    destroyNeuralNetwork(&network);
    return 1;
  }
  error = clEnqueueWriteBuffer(network.queue, network.layers[2].biases, CL_TRUE,
                               0, sizeof(layer3_biases_data),
                               layer3_biases_data, 0, NULL, NULL);
  if (error != CL_SUCCESS) {
    destroyNeuralNetwork(&network);
    return 1;
  }

  // 2. Prepare input data
  float input_data[] = {0.5f, 0.8f};
  error =
      clEnqueueWriteBuffer(network.queue, network.layers[0].input, CL_TRUE, 0,
                           sizeof(input_data), input_data, 0, NULL, NULL);
  if (error != CL_SUCCESS) {
    destroyNeuralNetwork(&network);
    return 1;
  }
  cl_mem input_buffer = network.layers[0].input; // First layer input

  // 3. Forward propagation
  for (int i = 0; i < num_layers - 1; i++) {
    if (forwardLayer(&network, &network.layers[i], input_buffer) != 0) {
      fprintf(stderr, "Forward pass failed at layer %d\n", i);
      destroyNeuralNetwork(&network);
      return 1;
    }
    input_buffer =
        network.layers[i]
            .output; // Output of current layer becomes input for next
  }

  // 4. Get output
  float output_data[2]; // Output layer size is 2
  error = clEnqueueReadBuffer(network.queue,
                              network.layers[num_layers - 2].output, CL_TRUE, 0,
                              sizeof(output_data), output_data, 0, NULL, NULL);
  if (error != CL_SUCCESS) {
    destroyNeuralNetwork(&network);
    return 1;
  }

  // 5. Verify output (manually calculated expected output)
  //   calculations:
  //    l1_out = relu(in * l1_w + l1_b)
  //    l2_out = relu(l1_out * l2_w + l2_b)
  //    l3_out = softmax(l2_out * l3_w + l3_b)
  //
  float expected_output[2];
  float l1_out[3];
  float l2_out[4];

  // Layer 1
  l1_out[0] = fmaxf(0.0f, input_data[0] * layer1_weights_data[0] +
                              input_data[1] * layer1_weights_data[1] +
                              layer1_biases_data[0]);
  l1_out[1] = fmaxf(0.0f, input_data[0] * layer1_weights_data[2] +
                              input_data[1] * layer1_weights_data[3] +
                              layer1_biases_data[1]);
  l1_out[2] = fmaxf(0.0f, input_data[0] * layer1_weights_data[4] +
                              input_data[1] * layer1_weights_data[5] +
                              layer1_biases_data[2]);

  // Layer 2
  l2_out[0] = fmaxf(0.0f, l1_out[0] * layer2_weights_data[0] +
                              l1_out[1] * layer2_weights_data[1] +
                              l1_out[2] * layer2_weights_data[2] +
                              layer2_biases_data[0]);
  l2_out[1] = fmaxf(0.0f, l1_out[0] * layer2_weights_data[3] +
                              l1_out[1] * layer2_weights_data[4] +
                              l1_out[2] * layer2_weights_data[5] +
                              layer2_biases_data[1]);
  l2_out[2] = fmaxf(0.0f, l1_out[0] * layer2_weights_data[6] +
                              l1_out[1] * layer2_weights_data[7] +
                              l1_out[2] * layer2_weights_data[8] +
                              layer2_biases_data[2]);
  l2_out[3] = fmaxf(0.0f, l1_out[0] * layer2_weights_data[9] +
                              l1_out[1] * layer2_weights_data[10] +
                              l1_out[2] * layer2_weights_data[11] +
                              layer2_biases_data[3]);

  // Layer 3 (Softmax)
  float max_val =
      fmaxf(fmaxf(l2_out[0], l2_out[1]), fmaxf(l2_out[2], l2_out[3]));
  float exp_sum = expf(l2_out[0] - max_val) + expf(l2_out[1] - max_val) +
                  expf(l2_out[2] - max_val) + expf(l2_out[3] - max_val);
  expected_output[0] = expf(l2_out[0] - max_val) / exp_sum;
  expected_output[1] = expf(l2_out[1] - max_val) / exp_sum;

  if (compareFloatArrays(expected_output, output_data, 2, 0.001f)) {
    printf("Forward propagation test passed!\n");
  } else {
    printf("Forward propagation test failed!\n");
    destroyNeuralNetwork(&network);
    return 1;
  }

  // 6. Backward Propagation Test (VERY basic - just checks no crash)
  // Prepare target data for backward pass.  In a real scenario, this would
  // come from your training data.  For this test, we'll make up some target
  // values.
  float target_data[] = {1.0f, 0.0f};
  error = clEnqueueWriteBuffer(network.queue,
                               network.layers[num_layers - 2].input, CL_TRUE, 0,
                               sizeof(target_data), target_data, 0, NULL, NULL);
  if (error != CL_SUCCESS) {
    destroyNeuralNetwork(&network);
    return 1;
  }
  cl_mem target_buffer = network.layers[num_layers - 2].input;

  // Backpropagate through the layers.
  cl_mem next_layer_error =
      target_buffer; // Start with target, which will be overwritten.
  for (int i = num_layers - 2; i >= 0; i--) {
    // Need to pass the input of the layer
    cl_mem layer_input =
        (i > 0) ? network.layers[i - 1].output : network.layers[0].input;
    if (backwardLayer(&network, &network.layers[i], next_layer_error,
                      layer_input, learning_rate) != 0) {
      fprintf(stderr, "Backward pass failed at layer %d\n", i);
      destroyNeuralNetwork(&network);
      return 1;
    }
    next_layer_error = network.layers[i].input; // Get error for previous layer.
  }
  for (int i = 0; i < num_layers - 1; i++) {
    if (applyGradients(&network, &network.layers[i], learning_rate) != 0) {
      fprintf(stderr, "Apply Gradients failed at layer %d\n", i);
      destroyNeuralNetwork(&network);
      return 1;
    }
  }

  printf(
      "Backward propagation test completed (no crash)!\n"); // Very basic check.
                                                            // A more thorough
                                                            // test would check
                                                            // the updated
                                                            // weights/biases.

  // 7. Cleanup
  destroyNeuralNetwork(&network);
  return 0;
}
