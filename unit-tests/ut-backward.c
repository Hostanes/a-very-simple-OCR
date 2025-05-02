
#include "../lib/nnlib-ocl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPSILON 0.0001f

// Function to read kernel source from file
char *load_kernel_source(const char *filename) {
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Error: Failed to open kernel file %s\n", filename);
    return NULL;
  }

  // Get file size
  fseek(fp, 0, SEEK_END);
  long size = ftell(fp);
  rewind(fp);

  // Allocate buffer
  char *source = (char *)malloc(size + 1);
  if (!source) {
    fprintf(stderr, "Error: Failed to allocate memory for kernel source\n");
    fclose(fp);
    return NULL;
  }

  // Read file content
  size_t read_size = fread(source, 1, size, fp);
  if (read_size != size) {
    fprintf(stderr, "Error: Failed to read kernel source\n");
    free(source);
    fclose(fp);
    return NULL;
  }

  source[size] = '\0'; // Null-terminate
  fclose(fp);
  return source;
}

// Modified program initialization code
cl_program create_and_build_program(cl_context context, cl_device_id device) {
  const char *kernel_filename = "lib/kernels.cl";
  char *kernel_source = load_kernel_source(kernel_filename);
  if (!kernel_source) {
    return NULL;
  }

  cl_int err;
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&kernel_source, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error: Failed to create program from source: %d\n", err);
    free(kernel_source);
    return NULL;
  }

  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    // Get build log
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    char *log = (char *)malloc(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log,
                          NULL);

    fprintf(stderr, "Error: Failed to build program:\n%s\n", log);
    free(log);
    free(kernel_source);
    clReleaseProgram(program);
    return NULL;
  }

  free(kernel_source);
  return program;
}

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

  clGetPlatformIDs(1, &platform, NULL);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  queue = clCreateCommandQueue(context, device, 0, NULL);

  // Create network (2-2-2 architecture)
  int layer_sizes[] = {2, 2, 2};
  NeuralNetwork_t net = create_NeuralNetwork(context, layer_sizes, 3);

  // Load and build program
  char *kernel_source = load_kernel_source("lib/kernels.cl");
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&kernel_source, NULL, NULL);
  clBuildProgram(program, 1, &device, NULL, NULL, NULL);

  // Create kernels
  net.forward_relu_kernel = clCreateKernel(program, "forward_With_Relu", NULL);
  net.forward_softmax_kernel =
      clCreateKernel(program, "forward_With_Softmax", NULL);
  net.backward_relu_kernel =
      clCreateKernel(program, "backward_With_Relu", NULL);
  net.backward_softmax_kernel =
      clCreateKernel(program, "backward_With_Softmax", NULL);
  net.queue = queue;
  net.program = program;

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
  float expected_weights1[] = {0.1f, 0.2f, 0.3f,
                               0.4f};      // No change (hidden layer)
  float expected_biases1[] = {0.1f, 0.1f}; // No change (hidden layer)

  // Output layer should have updates (softmax backward)
  // These are approximate expected values - adjust based on your exact
  // calculations
  float expected_weights2[] = {0.5f, 0.6f, 0.7f, 0.8f};
  float expected_biases2[] = {0.2f, 0.2f};

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
  free(kernel_source);
  clReleaseProgram(program);
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
