#include "../lib/nnlib-ocl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EPSILON 0.0001f

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper function to compare floats
int float_equal(float a, float b) { return fabs(a - b) < EPSILON; }

// Helper function to print array
void print_array(const char *name, float *arr, int size) {
  printf("%s: [", name);
  for (int i = 0; i < size; i++) {
    printf("%.4f", arr[i]);
    if (i < size - 1)
      printf(", ");
  }
  printf("]\n");
}

int main() {
  // Initialize OpenCL
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_int err;

  err = clGetPlatformIDs(1, &platform, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error: Failed to find OpenCL platform.\n");
    return 1;
  }

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error: Failed to find a GPU device.\n");
    return 1;
  }

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error: Failed to create OpenCL context.\n");
    return 1;
  }

  queue = clCreateCommandQueue(context, device, 0, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error: Failed to create command queue.\n");
    clReleaseContext(context);
    return 1;
  }

  // Define network architecture: input(3) -> hidden(3) -> output(3)
  int layer_sizes[] = {3, 3, 3};
  int num_layers = sizeof(layer_sizes) / sizeof(int);

  // Create network
  NeuralNetwork_t net =
      create_NeuralNetwork(context, device, queue, layer_sizes, num_layers);
  if (net.program == NULL) {
    fprintf(stderr, "Error: Failed to create neural network.\n");
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  // Manually set weights and biases for testing
  // Hidden layer weights (3x3 matrix)
  float hidden_weights[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                            0.6f, 0.7f, 0.8f, 0.9f};
  float hidden_biases[] = {0.1f, 0.1f, 0.1f};

  // Output layer weights (3x3 matrix)
  float output_weights[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                            0.6f, 0.7f, 0.8f, 0.9f};
  float output_biases[] = {0.1f, 0.1f, 0.1f};

  // Write weights and biases to device
  clEnqueueWriteBuffer(queue, net.layers[0].weights, CL_TRUE, 0,
                       9 * sizeof(float), hidden_weights, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, net.layers[0].biases, CL_TRUE, 0,
                       3 * sizeof(float), hidden_biases, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, net.layers[1].weights, CL_TRUE, 0,
                       9 * sizeof(float), output_weights, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, net.layers[1].biases, CL_TRUE, 0,
                       3 * sizeof(float), output_biases, 0, NULL, NULL);

  // Test input
  float input[] = {1.0f, 2.0f, 3.0f};
  float expected_hidden_output[] = {
      fmaxf(0.0f, 1.0f * 0.1f + 2.0f * 0.4f + 3.0f * 0.7f + 0.1f),
      fmaxf(0.0f, 1.0f * 0.2f + 2.0f * 0.5f + 3.0f * 0.8f + 0.1f),
      fmaxf(0.0f, 1.0f * 0.3f + 2.0f * 0.6f + 3.0f * 0.9f + 0.1f)};

  // Calculate expected final output (softmax)
  float output_sums[3];
  for (int i = 0; i < 3; i++) {
    output_sums[i] = expected_hidden_output[0] * output_weights[i] +
                     expected_hidden_output[1] * output_weights[3 + i] +
                     expected_hidden_output[2] * output_weights[6 + i] +
                     output_biases[i];
  }

  // Softmax calculation
  float max_val = fmaxf(fmaxf(output_sums[0], output_sums[1]), output_sums[2]);
  float exp_sum = expf(output_sums[0] - max_val) +
                  expf(output_sums[1] - max_val) +
                  expf(output_sums[2] - max_val);
  float expected_final_output[3];
  for (int i = 0; i < 3; i++) {
    expected_final_output[i] = expf(output_sums[i] - max_val) / exp_sum;
  }

  // Run forward pass
  float actual_output[3];
  forward(&net, queue, input, 3, actual_output, 3);

  // Read hidden layer output for verification
  float actual_hidden_output[3];
  clEnqueueReadBuffer(queue, net.layers[0].activation, CL_TRUE, 0,
                      3 * sizeof(float), actual_hidden_output, 0, NULL, NULL);

  // Verify results
  int passed = 1;

  // Check hidden layer output
  printf("Testing hidden layer output...\n");
  print_array("Expected", expected_hidden_output, 3);
  print_array("Actual", actual_hidden_output, 3);

  for (int i = 0; i < 3; i++) {
    if (!float_equal(actual_hidden_output[i], expected_hidden_output[i])) {
      printf("Mismatch at hidden output %d: expected %.4f, got %.4f\n", i,
             expected_hidden_output[i], actual_hidden_output[i]);
      passed = 0;
    }
  }

  // Check final output
  printf("\nTesting final output...\n");
  print_array("Expected", expected_final_output, 3);
  print_array("Actual", actual_output, 3);

  for (int i = 0; i < 3; i++) {
    if (!float_equal(actual_output[i], expected_final_output[i])) {
      printf("Mismatch at output %d: expected %.4f, got %.4f\n", i,
             expected_final_output[i], actual_output[i]);
      passed = 0;
    }
  }

  // Check that outputs sum to ~1.0 (softmax property)
  float sum = actual_output[0] + actual_output[1] + actual_output[2];
  if (!float_equal(sum, 1.0f)) {
    printf("Softmax outputs don't sum to 1.0: got %.4f\n", sum);
    passed = 0;
  }

  // Print test result
  if (passed) {
    printf("\nTEST PASSED!\n");
  } else {
    printf("\nTEST FAILED!\n");
  }

  // Cleanup
  free_NeuralNetwork(&net);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return passed ? 0 : 1;
}
