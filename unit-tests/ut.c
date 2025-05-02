
#include "../lib/nnlib-ocl.h"
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EPSILON 1e-6

// Function to compare two float arrays with tolerance
int compare_arrays(const float *a, const float *b, int size, float tolerance) {
  for (int i = 0; i < size; i++) {
    if (fabs(a[i] - b[i]) > tolerance) {
      printf("Mismatch at index %d: %f vs %f\n", i, a[i], b[i]);
      return 0;
    }
  }
  return 1;
}

// Function to initialize OpenCL
int initialize_opencl(cl_context *context, cl_device_id *device,
                      cl_command_queue *queue) {
  cl_int err;
  cl_platform_id platform;

  // Get platform
  err = clGetPlatformIDs(1, &platform, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error getting platform ID: %d\n", err);
    return 0;
  }

  // Get device (GPU preferred)
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, device, NULL);
  if (err == CL_DEVICE_NOT_FOUND) {
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, device, NULL);
  }
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error getting device ID: %d\n", err);
    return 0;
  }

  // Create context
  *context = clCreateContext(NULL, 1, device, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error creating context: %d\n", err);
    return 0;
  }

  // Create command queue
  *queue = clCreateCommandQueueWithProperties(*context, *device, 0, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error creating command queue: %d\n", err);
    clReleaseContext(*context);
    return 0;
  }

  return 1;
}

void test_4layer_network() {
  printf("Starting 4-layer neural network test...\n");

  // Initialize OpenCL
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;

  if (!initialize_opencl(&context, &device, &queue)) {
    fprintf(stderr, "Failed to initialize OpenCL\n");
    return;
  }

  // Define network architecture (4 layers: 2-3-3-2)
  int layer_sizes[] = {2, 3, 3, 2};
  int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);

  // Create neural network
  NeuralNetwork_t net =
      create_NeuralNetwork(context, device, queue, layer_sizes, num_layers);

  // Initialize weights and biases with known values for testing
  // Layer 1 (2x3)
  float layer1_weights[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
  float layer1_biases[] = {0.1f, 0.2f, 0.3f};

  // Layer 2 (3x3)
  float layer2_weights[] = {0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                            0.7f, 0.8f, 0.9f, 1.0f};
  float layer2_biases[] = {0.2f, 0.3f, 0.4f};

  // Layer 3 (3x2)
  float layer3_weights[] = {0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  float layer3_biases[] = {0.3f, 0.4f};

  // Write weights and biases to device
  clEnqueueWriteBuffer(queue, net.layers[0].weights, CL_TRUE, 0,
                       layer_sizes[0] * layer_sizes[1] * sizeof(float),
                       layer1_weights, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, net.layers[0].biases, CL_TRUE, 0,
                       layer_sizes[1] * sizeof(float), layer1_biases, 0, NULL,
                       NULL);

  clEnqueueWriteBuffer(queue, net.layers[1].weights, CL_TRUE, 0,
                       layer_sizes[1] * layer_sizes[2] * sizeof(float),
                       layer2_weights, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, net.layers[1].biases, CL_TRUE, 0,
                       layer_sizes[2] * sizeof(float), layer2_biases, 0, NULL,
                       NULL);

  clEnqueueWriteBuffer(queue, net.layers[2].weights, CL_TRUE, 0,
                       layer_sizes[2] * layer_sizes[3] * sizeof(float),
                       layer3_weights, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, net.layers[2].biases, CL_TRUE, 0,
                       layer_sizes[3] * sizeof(float), layer3_biases, 0, NULL,
                       NULL);

  // Test input and expected output
  float input[] = {0.5f, 0.8f};
  float target[] = {0.9f, 0.1f};

  // Expected activations (manually calculated)
  // Layer 1 (ReLU):
  //   Neuron 1: max(0, 0.5*0.1 + 0.8*0.4 + 0.1) = max(0, 0.47) = 0.47
  //   Neuron 2: max(0, 0.5*0.2 + 0.8*0.5 + 0.2) = max(0, 0.70) = 0.70
  //   Neuron 3: max(0, 0.5*0.3 + 0.8*0.6 + 0.3) = max(0, 0.93) = 0.93
  float expected_layer1_activation[] = {0.47f, 0.70f, 0.93f};

  // Layer 2 (ReLU):
  //   Neuron 1: max(0, 0.47*0.2 + 0.70*0.5 + 0.93*0.8 + 0.2) = max(0, 1.334)
  //   = 1.334 Neuron 2: max(0, 0.47*0.3 + 0.70*0.6 + 0.93*0.9 + 0.3) =
  //   max(0, 1.587) = 1.587 Neuron 3: max(0, 0.47*0.4 + 0.70*0.7 + 0.93*1.0 +
  //   0.4) = max(0, 1.858) = 1.858
  float expected_layer2_activation[] = {1.334f, 1.587f, 1.858f};

  // Layer 3 (Softmax):
  //   First compute unnormalized outputs:
  //     Neuron 1: 1.334*0.3 + 1.587*0.5 + 1.858*0.7 + 0.3 = 3.0927
  //     Neuron 2: 1.334*0.4 + 1.587*0.6 + 1.858*0.8 + 0.4 = 3.3508
  //   Then softmax:
  //     sum = exp(3.0927) + exp(3.3508) ≈ 22.0856 + 28.5133 ≈ 50.5989
  //     output1 = exp(3.0927)/sum ≈ 0.4365
  //     output2 = exp(3.3508)/sum ≈ 0.5635
  float expected_output[] = {0.4365f, 0.5635f};

  // Test forward pass
  printf("Testing forward pass...\n");
  float output[2];
  forward(&net, queue, input, 2, output, 2);

  // Verify output
  if (!compare_arrays(output, expected_output, 2, EPSILON)) {
    printf("Forward pass test failed at final output!\n");
  } else {
    printf("Final output matches expected values!\n");
  }

  // Verify intermediate activations (layer 1)
  float layer1_activation[3];
  clEnqueueReadBuffer(queue, net.layers[0].activation, CL_TRUE, 0,
                      3 * sizeof(float), layer1_activation, 0, NULL, NULL);
  if (!compare_arrays(layer1_activation, expected_layer1_activation, 3,
                      EPSILON)) {
    printf("Forward pass test failed at layer 1 activation!\n");
  } else {
    printf("Layer 1 activation matches expected values!\n");
  }

  // Verify intermediate activations (layer 2)
  float layer2_activation[3];
  clEnqueueReadBuffer(queue, net.layers[1].activation, CL_TRUE, 0,
                      3 * sizeof(float), layer2_activation, 0, NULL, NULL);
  if (!compare_arrays(layer2_activation, expected_layer2_activation, 3,
                      EPSILON)) {
    printf("Forward pass test failed at layer 2 activation!\n");
  } else {
    printf("Layer 2 activation matches expected values!\n");
  }

  // Test backward pass
  printf("\nTesting backward pass...\n");
  float learning_rate = 0.01f;
  float momentum = 0.9f;

  // Perform training step (forward + backward)
  train(&net, queue, input, 2, target, 2, learning_rate, momentum);

  // Verify gradients and weight updates
  // For this test, we'll just check that weights have changed (not zero delta)

  // Read back weights after update
  float updated_layer1_weights[6];
  float updated_layer2_weights[9];
  float updated_layer3_weights[6];

  clEnqueueReadBuffer(queue, net.layers[0].weights, CL_TRUE, 0,
                      6 * sizeof(float), updated_layer1_weights, 0, NULL, NULL);
  clEnqueueReadBuffer(queue, net.layers[1].weights, CL_TRUE, 0,
                      9 * sizeof(float), updated_layer2_weights, 0, NULL, NULL);
  clEnqueueReadBuffer(queue, net.layers[2].weights, CL_TRUE, 0,
                      6 * sizeof(float), updated_layer3_weights, 0, NULL, NULL);

  // Check if weights have changed
  int weights_changed = 0;
  for (int i = 0; i < 6; i++) {
    if (fabs(updated_layer1_weights[i] - layer1_weights[i]) > EPSILON) {
      weights_changed = 1;
      break;
    }
  }

  if (!weights_changed) {
    printf("Layer 1 weights not updated during backward pass!\n");
  } else {
    printf("Layer 1 weights updated correctly.\n");
  }

  weights_changed = 0;
  for (int i = 0; i < 9; i++) {
    if (fabs(updated_layer2_weights[i] - layer2_weights[i]) > EPSILON) {
      weights_changed = 1;
      break;
    }
  }

  if (!weights_changed) {
    printf("Layer 2 weights not updated during backward pass!\n");
  } else {
    printf("Layer 2 weights updated correctly.\n");
  }

  weights_changed = 0;
  for (int i = 0; i < 6; i++) {
    if (fabs(updated_layer3_weights[i] - layer3_weights[i]) > EPSILON) {
      weights_changed = 1;
      break;
    }
  }

  if (!weights_changed) {
    printf("Layer 3 weights not updated during backward pass!\n");
  } else {
    printf("Layer 3 weights updated correctly.\n");
  }

  // Cleanup
  free_NeuralNetwork(&net);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  printf("\n4-layer neural network test completed.\n");
}

int main() {
  test_4layer_network();
  return 0;
}
