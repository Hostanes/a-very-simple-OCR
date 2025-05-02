
#include "../lib/nnlib-ocl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EPSILON 0.0001f

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

  clGetPlatformIDs(1, &platform, NULL);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  queue = clCreateCommandQueue(context, device, 0, NULL);

  // Define network architecture: input(3) -> hidden(3) -> output(3)
  int layer_sizes[] = {3, 3, 3};
  int num_layers = sizeof(layer_sizes) / sizeof(int);

  // Create network
  NeuralNetwork_t net = create_NeuralNetwork(context, layer_sizes, num_layers);

  // Load and build OpenCL program
  cl_program program = create_and_build_program(context, device);
  if (!program) {
    // Cleanup and exit if program creation fails
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }

  // Assign to network structure
  net.program = program;

  // Create kernels
  net.forward_relu_kernel = clCreateKernel(program, "forward_With_Relu", NULL);
  net.forward_softmax_kernel =
      clCreateKernel(program, "forward_With_Softmax", NULL);
  net.queue = queue;
  net.program = program;

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
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return passed ? 0 : 1;
}
