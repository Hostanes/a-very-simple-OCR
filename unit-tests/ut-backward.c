#include "../lib/nnlib-ocl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPSILON 0.0001f

// Helper functions
int float_equal(float a, float b) { return fabs(a - b) < EPSILON; }

void print_array(const char *name, float *arr, int size) {
  printf("%s: [", name);
  for (int i = 0; i < size; i++) {
    printf("%.4f", arr[i]);
    if (i < size - 1)
      printf(", ");
  }
  printf("]\n");
}

void init_network_with_test_values(NeuralNetwork_t *net,
                                   cl_command_queue queue) {
  // Simple 2-layer network: 2 inputs -> 2 hidden -> 2 outputs
  float weights1[] = {0.1f, 0.2f, 0.3f, 0.4f}; // 2x2 matrix
  float biases1[] = {0.1f, 0.1f};
  float weights2[] = {0.5f, 0.6f, 0.7f, 0.8f}; // 2x2 matrix
  float biases2[] = {0.2f, 0.2f};

  // Write weights and biases to device
  Layer_t *layer1 = &net->layers[0];
  clEnqueueWriteBuffer(queue, layer1->weights, CL_TRUE, 0, 4 * sizeof(float),
                       weights1, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, layer1->biases, CL_TRUE, 0, 2 * sizeof(float),
                       biases1, 0, NULL, NULL);

  Layer_t *layer2 = &net->layers[1];
  clEnqueueWriteBuffer(queue, layer2->weights, CL_TRUE, 0, 4 * sizeof(float),
                       weights2, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, layer2->biases, CL_TRUE, 0, 2 * sizeof(float),
                       biases2, 0, NULL, NULL);
}

int test_backward_pass() {
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

  // Create network (2-2-2 architecture)
  int layer_sizes[] = {2, 2, 2};
  NeuralNetwork_t net =
      create_NeuralNetwork(context, device, queue, layer_sizes, 3);
  if (net.program == NULL) {
    fprintf(stderr, "Error: Failed to create neural network.\n");
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }

  // Initialize with test values
  init_network_with_test_values(&net, queue);

  // Test input and target
  float input[] = {1.0f, 0.5f};
  float target[] = {0.0f, 1.0f}; // One-hot target

  // Run forward pass first
  forward(&net, queue, input, 2, NULL, 0);

  // Run backward pass
  float learning_rate = 0.1f;
  float momentum = 0.9f;
  backward(&net, queue, target, 2, learning_rate, momentum);

  // Read back results for verification
  float updated_weights1[4], updated_biases1[2];
  float updated_weights2[4], updated_biases2[2];

  clEnqueueReadBuffer(queue, net.layers[0].weights, CL_TRUE, 0,
                      4 * sizeof(float), updated_weights1, 0, NULL, NULL);
  clEnqueueReadBuffer(queue, net.layers[0].biases, CL_TRUE, 0,
                      2 * sizeof(float), updated_biases1, 0, NULL, NULL);
  clEnqueueReadBuffer(queue, net.layers[1].weights, CL_TRUE, 0,
                      4 * sizeof(float), updated_weights2, 0, NULL, NULL);
  clEnqueueReadBuffer(queue, net.layers[1].biases, CL_TRUE, 0,
                      2 * sizeof(float), updated_biases2, 0, NULL, NULL);

  // Expected values (manually calculated)
  //  Forward Pass:
  //    input = [1.0, 0.5]
  //    h1_out = relu(1.0*0.1 + 0.5*0.3 + 0.1, 1.0*0.2 + 0.5*0.4 + 0.1) = [0.25,
  //    0.4] output_sums = [0.25*0.5 + 0.4*0.7 + 0.2, 0.25*0.6 + 0.4*0.8 + 0.2]
  //    = [0.505, 0.67] output = softmax([0.505, 0.67]) = [0.46, 0.54]
  //    (approximately)

  // Backward Pass:
  // dL/dOut = output - target = [0.46, 0.54] - [0, 1] = [0.46, -0.46]
  // For output layer (Softmax):
  //    dL/dW2 =  [[0.25],[0.4]] * [0.46, -0.46] = [[0.115, -0.115],[0.184,
  //    -0.184]] dL/dB2 = [0.46, -0.46]
  // For hidden layer (Relu):
  //  back_grad = dL/dOut * W2.T = [0.46, -0.46] * [[0.5, 0.6],[0.7,0.8]] =
  //  [-0.092, -0.046] relu_grad = back_grad * relu_derivative = [-0.092,
  //  -0.046] * [1, 1] = [-0.092, -0.046] dL/dW1 = [[1.0],[0.5]] * [-0.092,
  //  -0.046] = [[-0.092, -0.046],[-0.046, -0.023]] dL/dB1 = [-0.092, -0.046]

  float expected_weights2[] = {0.5f - learning_rate * 0.25f * 0.46f,  // w11
                               0.6f - learning_rate * 0.25f * -0.46f, // w12
                               0.7f - learning_rate * 0.4f * 0.46f,   // w21
                               0.8f - learning_rate * 0.4f * -0.46f}; // w22
  float expected_biases2[] = {0.2f - learning_rate * 0.46f,
                              0.2f - learning_rate * -0.46f};

  float expected_weights1[] = {0.1f, 0.2f, 0.3f, 0.4f};
  float expected_biases1[] = {0.1f, 0.1f};

  // Verify results
  int passed = 1;

  printf("\nTesting output layer updates (should change):\n");
  print_array("Expected weights", expected_weights2, 4);
  print_array("Actual weights", updated_weights2, 4);
  print_array("Expected biases", expected_biases2, 2);
  print_array("Actual biases", updated_biases2, 2);

  for (int i = 0; i < 4; i++) {
    if (!float_equal(updated_weights2[i], expected_weights2[i])) {
      printf("Weight mismatch at %d: expected %.4f, got %.4f\n", i,
             expected_weights2[i], updated_weights2[i]);
      passed = 0;
    }
  }

  for (int i = 0; i < 2; i++) {
    if (!float_equal(updated_biases2[i], expected_biases2[i])) {
      printf("Bias mismatch at %d: expected %.4f, got %.4f\n", i,
             expected_biases2[i], updated_biases2[i]);
      passed = 0;
    }
  }

  printf("\nTesting hidden layer updates (should NOT change):\n");
  print_array("Expected weights", expected_weights1, 4);
  print_array("Actual weights", updated_weights1, 4);
  print_array("Expected biases", expected_biases1, 2);
  print_array("Actual biases", updated_biases1, 2);

  for (int i = 0; i < 4; i++) {
    if (!float_equal(updated_weights1[i], expected_weights1[i])) {
      printf("Weight mismatch at %d: expected %.4f, got %.4f\n", i,
             expected_weights1[i], updated_weights1[i]);
      passed = 0;
    }
  }

  for (int i = 0; i < 2; i++) {
    if (!float_equal(updated_biases1[i], expected_biases1[i])) {
      printf("Bias mismatch at %d: expected %.4f, got %.4f\n", i,
             expected_biases1[i], updated_biases1[i]);
      passed = 0;
    }
  }

  // Cleanup
  free_NeuralNetwork(&net);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return passed;
}

int main() {
  printf("Running backward pass unit test...\n");

  if (test_backward_pass()) {
    printf("\nTEST PASSED!\n");
    return 0;
  } else {
    printf("\nTEST FAILED!\n");
    return 1;
  }
}
