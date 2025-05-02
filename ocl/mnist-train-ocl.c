#include <omp.h>
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#define MAGIC_NUMBER 143 // 0x8f

#define FILE_VERSION 1
#define MAX_ACTIVATION_NAME_LEN 32


typedef enum {
  ACT_RELU,
  ACT_SIGMOID,
  ACT_TANH,
  ACT_LINEAR,
  ACT_SOFTMAX
} ActivationType;



#define INPUT_SIZE 784
#define HIDDEN1_SIZE 512
#define HIDDEN2_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.0005f
#define MOMENTUM 0.9f
#define EPOCHS 4
#define BATCH_SIZE 64
#define IMAGE_SIZE 28
#define TRAIN_SPLIT 0.8
#define PRINT_INTERVAL 1000

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel forward_kernel, softmax_kernel, backward_output_kernel,
    hidden_gradients_kernel, backward_hidden_kernel;
cl_mem d_hidden1_weights, d_hidden1_biases, d_hidden1_weight_momentum,
    d_hidden1_bias_momentum;
cl_mem d_hidden2_weights, d_hidden2_biases, d_hidden2_weight_momentum,
    d_hidden2_bias_momentum;
cl_mem d_output_weights, d_output_biases, d_output_weight_momentum,
    d_output_bias_momentum;
cl_mem d_input, d_hidden1_output, d_hidden2_output, d_output, d_output_grad,
    d_hidden2_grad, d_hidden1_grad;

typedef struct {
  float *weights, *biases, *weight_momentum, *bias_momentum;
  int input_size, output_size;
} Layer;

typedef struct {
  Layer hidden1, hidden2, output;
} Network;

typedef struct {
  unsigned char *images, *labels;
  int nImages;
} InputData;

void check_error(cl_int err, const char *operation) {
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error during operation '%s': %d\n", operation, err);
    exit(EXIT_FAILURE);
  }
}

char *read_kernel_source(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "Failed to open kernel file: %s\n", filename);
    exit(EXIT_FAILURE);
  }

  fseek(file, 0, SEEK_END);
  size_t size = ftell(file);
  rewind(file);

  char *source = (char *)malloc(size + 1);
  if (!source) {
    fprintf(stderr, "Failed to allocate memory for kernel source\n");
    fclose(file);
    exit(EXIT_FAILURE);
  }

  fread(source, 1, size, file);
  source[size] = '\0';
  fclose(file);

  return source;
}

void init_opencl() {
  cl_int err;

  err = clGetPlatformIDs(1, &platform, NULL);
  check_error(err, "Getting platform ID");

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (err != CL_SUCCESS) {
    printf("GPU device not found, trying CPU...\n");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    check_error(err, "Getting device ID");
  }


  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  check_error(err, "Creating context");

  queue = clCreateCommandQueue(context, device, 0, &err);
  check_error(err, "Creating command queue");

  const char *source = read_kernel_source("ocl/kernels.cl");
  program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
  check_error(err, "Creating program");

  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    char *log = (char *)malloc(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log,
                          NULL);
    fprintf(stderr, "Program build error: %s\n", log);
    free(log);
    exit(EXIT_FAILURE);
  }

  forward_kernel = clCreateKernel(program, "forward_pass", &err);
  check_error(err, "Creating forward kernel");

  softmax_kernel = clCreateKernel(program, "softmax", &err);
  check_error(err, "Creating softmax kernel");

  backward_output_kernel = clCreateKernel(program, "backward_output", &err);
  check_error(err, "Creating backward_output kernel");

  hidden_gradients_kernel = clCreateKernel(program, "hidden_gradients", &err);
  check_error(err, "Creating hidden_gradients kernel");

  backward_hidden_kernel = clCreateKernel(program, "backward_hidden", &err);
  check_error(err, "Creating backward_hidden kernel");

  free((void *)source);
}

void create_opencl_buffers(Network *net) {
  cl_int err;

  // Hidden1 layer buffers
  d_hidden1_weights = clCreateBuffer(
      context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(float) * INPUT_SIZE * HIDDEN1_SIZE, net->hidden1.weights, &err);
  check_error(err, "Creating hidden1 weights buffer");

  d_hidden1_biases =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * HIDDEN1_SIZE, net->hidden1.biases, &err);
  check_error(err, "Creating hidden1 biases buffer");

  d_hidden1_weight_momentum =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * INPUT_SIZE * HIDDEN1_SIZE,
                     net->hidden1.weight_momentum, &err);
  check_error(err, "Creating hidden1 weight momentum buffer");

  d_hidden1_bias_momentum = clCreateBuffer(
      context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(float) * HIDDEN1_SIZE, net->hidden1.bias_momentum, &err);
  check_error(err, "Creating hidden1 bias momentum buffer");

  // Hidden2 layer buffers
  d_hidden2_weights = clCreateBuffer(
      context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(float) * HIDDEN1_SIZE * HIDDEN2_SIZE, net->hidden2.weights, &err);
  check_error(err, "Creating hidden2 weights buffer");

  d_hidden2_biases =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * HIDDEN2_SIZE, net->hidden2.biases, &err);
  check_error(err, "Creating hidden2 biases buffer");

  d_hidden2_weight_momentum =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * HIDDEN1_SIZE * HIDDEN2_SIZE,
                     net->hidden2.weight_momentum, &err);
  check_error(err, "Creating hidden2 weight momentum buffer");

  d_hidden2_bias_momentum = clCreateBuffer(
      context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(float) * HIDDEN2_SIZE, net->hidden2.bias_momentum, &err);
  check_error(err, "Creating hidden2 bias momentum buffer");

  // Output layer buffers
  d_output_weights = clCreateBuffer(
      context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(float) * HIDDEN2_SIZE * OUTPUT_SIZE, net->output.weights, &err);
  check_error(err, "Creating output weights buffer");

  d_output_biases =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * OUTPUT_SIZE, net->output.biases, &err);
  check_error(err, "Creating output biases buffer");

  d_output_weight_momentum =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * HIDDEN2_SIZE * OUTPUT_SIZE,
                     net->output.weight_momentum, &err);
  check_error(err, "Creating output weight momentum buffer");

  d_output_bias_momentum = clCreateBuffer(
      context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(float) * OUTPUT_SIZE, net->output.bias_momentum, &err);
  check_error(err, "Creating output bias momentum buffer");

  // Data buffers
  d_input = clCreateBuffer(context, CL_MEM_READ_WRITE,
                           sizeof(float) * INPUT_SIZE, NULL, &err);
  check_error(err, "Creating input buffer");

  d_hidden1_output = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * HIDDEN1_SIZE, NULL, &err);
  check_error(err, "Creating hidden1 output buffer");

  d_hidden2_output = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * HIDDEN2_SIZE, NULL, &err);
  check_error(err, "Creating hidden2 output buffer");

  d_output = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * OUTPUT_SIZE, NULL, &err);
  check_error(err, "Creating output buffer");

  d_output_grad = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(float) * OUTPUT_SIZE, NULL, &err);
  check_error(err, "Creating output gradient buffer");

  d_hidden2_grad = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  sizeof(float) * HIDDEN2_SIZE, NULL, &err);
  check_error(err, "Creating hidden2 gradient buffer");

  d_hidden1_grad = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  sizeof(float) * HIDDEN1_SIZE, NULL, &err);
  check_error(err, "Creating hidden1 gradient buffer");
}

void cleanup_opencl() {
  // Release hidden1 layer buffers
  clReleaseMemObject(d_hidden1_weights);
  clReleaseMemObject(d_hidden1_biases);
  clReleaseMemObject(d_hidden1_weight_momentum);
  clReleaseMemObject(d_hidden1_bias_momentum);

  // Release hidden2 layer buffers
  clReleaseMemObject(d_hidden2_weights);
  clReleaseMemObject(d_hidden2_biases);
  clReleaseMemObject(d_hidden2_weight_momentum);
  clReleaseMemObject(d_hidden2_bias_momentum);

  // Release output layer buffers
  clReleaseMemObject(d_output_weights);
  clReleaseMemObject(d_output_biases);
  clReleaseMemObject(d_output_weight_momentum);
  clReleaseMemObject(d_output_bias_momentum);

  // Release data buffers
  clReleaseMemObject(d_input);
  clReleaseMemObject(d_hidden1_output);
  clReleaseMemObject(d_hidden2_output);
  clReleaseMemObject(d_output);
  clReleaseMemObject(d_output_grad);
  clReleaseMemObject(d_hidden2_grad);
  clReleaseMemObject(d_hidden1_grad);

  // Release kernels
  clReleaseKernel(forward_kernel);
  clReleaseKernel(softmax_kernel);
  clReleaseKernel(backward_output_kernel);
  clReleaseKernel(hidden_gradients_kernel);
  clReleaseKernel(backward_hidden_kernel);

  // Release program and context
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}

void init_layer(Layer *layer, int in_size, int out_size) {
  int n = in_size * out_size;
  float scale = sqrtf(2.0f / in_size);

  layer->input_size = in_size;
  layer->output_size = out_size;
  layer->weights = malloc(n * sizeof(float));
  layer->biases = calloc(out_size, sizeof(float));
  layer->weight_momentum = calloc(n, sizeof(float));
  layer->bias_momentum = calloc(out_size, sizeof(float));

  for (int i = 0; i < n; i++)
    layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
}

void forward_opencl(float *input, float *hidden1_output, float *hidden2_output,
                    float *final_output) {
  cl_int err;
  size_t global_size, local_size;

  // First hidden layer (input -> hidden1)
  int input_size = INPUT_SIZE;
  int hidden1_size = HIDDEN1_SIZE;

  err = clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0,
                             sizeof(float) * INPUT_SIZE, input, 0, NULL, NULL);
  check_error(err, "Writing input buffer");

  err = clSetKernelArg(forward_kernel, 0, sizeof(int), &input_size);
  err |= clSetKernelArg(forward_kernel, 1, sizeof(int), &hidden1_size);
  err |= clSetKernelArg(forward_kernel, 2, sizeof(cl_mem), &d_hidden1_weights);
  err |= clSetKernelArg(forward_kernel, 3, sizeof(cl_mem), &d_hidden1_biases);
  err |= clSetKernelArg(forward_kernel, 4, sizeof(cl_mem), &d_input);
  err |= clSetKernelArg(forward_kernel, 5, sizeof(cl_mem), &d_hidden1_output);
  check_error(err, "Setting hidden1 layer forward kernel arguments");

  global_size = HIDDEN1_SIZE;
  local_size = 64;
  err = clEnqueueNDRangeKernel(queue, forward_kernel, 1, NULL, &global_size,
                               &local_size, 0, NULL, NULL);
  check_error(err, "Executing hidden1 layer forward kernel");

  // Second hidden layer (hidden1 -> hidden2)
  int hidden2_size = HIDDEN2_SIZE;

  err = clSetKernelArg(forward_kernel, 0, sizeof(int), &hidden1_size);
  err |= clSetKernelArg(forward_kernel, 1, sizeof(int), &hidden2_size);
  err |= clSetKernelArg(forward_kernel, 2, sizeof(cl_mem), &d_hidden2_weights);
  err |= clSetKernelArg(forward_kernel, 3, sizeof(cl_mem), &d_hidden2_biases);
  err |= clSetKernelArg(forward_kernel, 4, sizeof(cl_mem), &d_hidden1_output);
  err |= clSetKernelArg(forward_kernel, 5, sizeof(cl_mem), &d_hidden2_output);
  check_error(err, "Setting hidden2 layer forward kernel arguments");

  global_size = HIDDEN2_SIZE;
  local_size = 64;
  err = clEnqueueNDRangeKernel(queue, forward_kernel, 1, NULL, &global_size,
                               &local_size, 0, NULL, NULL);
  check_error(err, "Executing hidden2 layer forward kernel");

  // Output layer (hidden2 -> output)
  int output_size = OUTPUT_SIZE;

  err = clSetKernelArg(forward_kernel, 0, sizeof(int), &hidden2_size);
  err |= clSetKernelArg(forward_kernel, 1, sizeof(int), &output_size);
  err |= clSetKernelArg(forward_kernel, 2, sizeof(cl_mem), &d_output_weights);
  err |= clSetKernelArg(forward_kernel, 3, sizeof(cl_mem), &d_output_biases);
  err |= clSetKernelArg(forward_kernel, 4, sizeof(cl_mem), &d_hidden2_output);
  err |= clSetKernelArg(forward_kernel, 5, sizeof(cl_mem), &d_output);
  check_error(err, "Setting output layer forward kernel arguments");

  global_size = OUTPUT_SIZE;
  local_size = 10;
  err = clEnqueueNDRangeKernel(queue, forward_kernel, 1, NULL, &global_size,
                               &local_size, 0, NULL, NULL);
  check_error(err, "Executing output layer forward kernel");

  // Apply softmax to output
  err = clSetKernelArg(softmax_kernel, 0, sizeof(int), &output_size);
  err |= clSetKernelArg(softmax_kernel, 1, sizeof(cl_mem), &d_output);
  check_error(err, "Setting softmax kernel arguments");

  global_size = 1;
  err = clEnqueueNDRangeKernel(queue, softmax_kernel, 1, NULL, &global_size,
                               NULL, 0, NULL, NULL);
  check_error(err, "Executing softmax kernel");

  // Read back outputs if requested
  if (hidden1_output) {
    err = clEnqueueReadBuffer(queue, d_hidden1_output, CL_TRUE, 0,
                              sizeof(float) * HIDDEN1_SIZE, hidden1_output, 0,
                              NULL, NULL);
    check_error(err, "Reading hidden1 output");
  }

  if (hidden2_output) {
    err = clEnqueueReadBuffer(queue, d_hidden2_output, CL_TRUE, 0,
                              sizeof(float) * HIDDEN2_SIZE, hidden2_output, 0,
                              NULL, NULL);
    check_error(err, "Reading hidden2 output");
  }

  if (final_output) {
    err = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0,
                              sizeof(float) * OUTPUT_SIZE, final_output, 0,
                              NULL, NULL);
    check_error(err, "Reading final output");
  }
}

void backward_opencl(float *input, int label, float lr) {
  cl_int err;
  size_t global_size, local_size;
  float output_grad[OUTPUT_SIZE] = {0};

  int hidden1_size = HIDDEN1_SIZE;
  int hidden2_size = HIDDEN2_SIZE;
  int output_size = OUTPUT_SIZE;
  float momentum = MOMENTUM;

  // Get output for gradient calculation
  float final_output[OUTPUT_SIZE];
  err = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0,
                            sizeof(float) * OUTPUT_SIZE, final_output, 0, NULL,
                            NULL);
  check_error(err, "Reading output for gradient calculation");

  // Calculate output gradient
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    output_grad[i] = final_output[i] - (i == label);
  }

  err = clEnqueueWriteBuffer(queue, d_output_grad, CL_TRUE, 0,
                             sizeof(float) * OUTPUT_SIZE, output_grad, 0, NULL,
                             NULL);
  check_error(err, "Writing output gradient");

  // Backpropagate through output layer (hidden2 -> output)
  err = clSetKernelArg(backward_output_kernel, 0, sizeof(int), &hidden2_size);
  err |= clSetKernelArg(backward_output_kernel, 1, sizeof(int), &output_size);
  err |= clSetKernelArg(backward_output_kernel, 2, sizeof(cl_mem),
                        &d_hidden2_output);
  err |=
      clSetKernelArg(backward_output_kernel, 3, sizeof(cl_mem), &d_output_grad);
  err |= clSetKernelArg(backward_output_kernel, 4, sizeof(cl_mem),
                        &d_output_weights);
  err |= clSetKernelArg(backward_output_kernel, 5, sizeof(cl_mem),
                        &d_output_weight_momentum);
  err |= clSetKernelArg(backward_output_kernel, 6, sizeof(cl_mem),
                        &d_output_bias_momentum);
  err |= clSetKernelArg(backward_output_kernel, 7, sizeof(float), &lr);
  err |= clSetKernelArg(backward_output_kernel, 8, sizeof(float), &momentum);
  check_error(err, "Setting backward output kernel arguments");

  global_size = OUTPUT_SIZE;
  local_size = 10;
  err = clEnqueueNDRangeKernel(queue, backward_output_kernel, 1, NULL,
                               &global_size, &local_size, 0, NULL, NULL);
  check_error(err, "Executing backward output kernel");

  // Calculate hidden2 gradients
  err = clSetKernelArg(hidden_gradients_kernel, 0, sizeof(int), &hidden2_size);
  err |= clSetKernelArg(hidden_gradients_kernel, 1, sizeof(int), &output_size);
  err |= clSetKernelArg(hidden_gradients_kernel, 2, sizeof(cl_mem),
                        &d_hidden2_output);
  err |= clSetKernelArg(hidden_gradients_kernel, 3, sizeof(cl_mem),
                        &d_output_grad);
  err |= clSetKernelArg(hidden_gradients_kernel, 4, sizeof(cl_mem),
                        &d_output_weights);
  err |= clSetKernelArg(hidden_gradients_kernel, 5, sizeof(cl_mem),
                        &d_hidden2_grad);
  check_error(err, "Setting hidden2 gradients kernel arguments");

  global_size = HIDDEN2_SIZE;
  local_size = 64;
  err = clEnqueueNDRangeKernel(queue, hidden_gradients_kernel, 1, NULL,
                               &global_size, &local_size, 0, NULL, NULL);
  check_error(err, "Executing hidden2 gradients kernel");

  // Backpropagate through hidden2 layer (hidden1 -> hidden2)
  err = clSetKernelArg(backward_hidden_kernel, 0, sizeof(int), &hidden1_size);
  err |= clSetKernelArg(backward_hidden_kernel, 1, sizeof(int), &hidden2_size);
  err |= clSetKernelArg(backward_hidden_kernel, 2, sizeof(cl_mem),
                        &d_hidden1_output);
  err |= clSetKernelArg(backward_hidden_kernel, 3, sizeof(cl_mem),
                        &d_hidden2_grad);
  err |= clSetKernelArg(backward_hidden_kernel, 4, sizeof(cl_mem),
                        &d_hidden2_weights);
  err |= clSetKernelArg(backward_hidden_kernel, 5, sizeof(cl_mem),
                        &d_hidden2_weight_momentum);
  err |= clSetKernelArg(backward_hidden_kernel, 6, sizeof(cl_mem),
                        &d_hidden2_bias_momentum);
  err |= clSetKernelArg(backward_hidden_kernel, 7, sizeof(float), &lr);
  err |= clSetKernelArg(backward_hidden_kernel, 8, sizeof(float), &momentum);
  check_error(err, "Setting backward hidden2 kernel arguments");

  global_size = HIDDEN2_SIZE;
  local_size = 64;
  err = clEnqueueNDRangeKernel(queue, backward_hidden_kernel, 1, NULL,
                               &global_size, &local_size, 0, NULL, NULL);
  check_error(err, "Executing backward hidden2 kernel");

  // Calculate hidden1 gradients
  err = clSetKernelArg(hidden_gradients_kernel, 0, sizeof(int), &hidden1_size);
  err |= clSetKernelArg(hidden_gradients_kernel, 1, sizeof(int), &hidden2_size);
  err |= clSetKernelArg(hidden_gradients_kernel, 2, sizeof(cl_mem),
                        &d_hidden1_output);
  err |= clSetKernelArg(hidden_gradients_kernel, 3, sizeof(cl_mem),
                        &d_hidden2_grad);
  err |= clSetKernelArg(hidden_gradients_kernel, 4, sizeof(cl_mem),
                        &d_hidden2_weights);
  err |= clSetKernelArg(hidden_gradients_kernel, 5, sizeof(cl_mem),
                        &d_hidden1_grad);
  check_error(err, "Setting hidden1 gradients kernel arguments");

  global_size = HIDDEN1_SIZE;
  local_size = 64;
  err = clEnqueueNDRangeKernel(queue, hidden_gradients_kernel, 1, NULL,
                               &global_size, &local_size, 0, NULL, NULL);
  check_error(err, "Executing hidden1 gradients kernel");

  // Backpropagate through hidden1 layer (input -> hidden1)
  int input_size = INPUT_SIZE;

  err = clSetKernelArg(backward_hidden_kernel, 0, sizeof(int), &input_size);
  err |= clSetKernelArg(backward_hidden_kernel, 1, sizeof(int), &hidden1_size);
  err |= clSetKernelArg(backward_hidden_kernel, 2, sizeof(cl_mem), &d_input);
  err |= clSetKernelArg(backward_hidden_kernel, 3, sizeof(cl_mem),
                        &d_hidden1_grad);
  err |= clSetKernelArg(backward_hidden_kernel, 4, sizeof(cl_mem),
                        &d_hidden1_weights);
  err |= clSetKernelArg(backward_hidden_kernel, 5, sizeof(cl_mem),
                        &d_hidden1_weight_momentum);
  err |= clSetKernelArg(backward_hidden_kernel, 6, sizeof(cl_mem),
                        &d_hidden1_bias_momentum);
  err |= clSetKernelArg(backward_hidden_kernel, 7, sizeof(float), &lr);
  err |= clSetKernelArg(backward_hidden_kernel, 8, sizeof(float), &momentum);
  check_error(err, "Setting backward hidden1 kernel arguments");

  global_size = HIDDEN1_SIZE;
  local_size = 64;
  err = clEnqueueNDRangeKernel(queue, backward_hidden_kernel, 1, NULL,
                               &global_size, &local_size, 0, NULL, NULL);
  check_error(err, "Executing backward hidden1 kernel");
}

float *train_opencl(Network *net, float *input, int label, float lr) {
  static float final_output[OUTPUT_SIZE];
  float hidden1_output[HIDDEN1_SIZE], hidden2_output[HIDDEN2_SIZE];

  forward_opencl(input, hidden1_output, hidden2_output, final_output);

  backward_opencl(input, label, lr);

  return final_output;
}

int predict_opencl(float *input) {
  float final_output[OUTPUT_SIZE];

  forward_opencl(input, NULL, NULL, final_output);

  int max_index = 0;
  for (int i = 1; i < OUTPUT_SIZE; i++)
    if (final_output[i] > final_output[max_index])
      max_index = i;

  return max_index;
}

void read_mnist_images(const char *filename, unsigned char **images,
                       int *nImages) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Failed to open file: %s\n", filename);
    exit(1);
  }

  int temp, rows, cols;
  fread(&temp, sizeof(int), 1, file);
  fread(nImages, sizeof(int), 1, file);
  *nImages = __builtin_bswap32(*nImages);

  fread(&rows, sizeof(int), 1, file);
  fread(&cols, sizeof(int), 1, file);

  rows = __builtin_bswap32(rows);
  cols = __builtin_bswap32(cols);

  *images = malloc((*nImages) * IMAGE_SIZE * IMAGE_SIZE);
  fread(*images, sizeof(unsigned char), (*nImages) * IMAGE_SIZE * IMAGE_SIZE,
        file);
  fclose(file);
}

void read_mnist_labels(const char *filename, unsigned char **labels,
                       int *nLabels) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Failed to open file: %s\n", filename);
    exit(1);
  }

  int temp;
  fread(&temp, sizeof(int), 1, file);
  fread(nLabels, sizeof(int), 1, file);
  *nLabels = __builtin_bswap32(*nLabels);

  *labels = malloc(*nLabels);
  fread(*labels, sizeof(unsigned char), *nLabels, file);
  fclose(file);
}

void shuffle_data(unsigned char *images, unsigned char *labels, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    for (int k = 0; k < INPUT_SIZE; k++) {
      unsigned char temp = images[i * INPUT_SIZE + k];
      images[i * INPUT_SIZE + k] = images[j * INPUT_SIZE + k];
      images[j * INPUT_SIZE + k] = temp;
    }
    unsigned char temp = labels[i];
    labels[i] = labels[j];
    labels[j] = temp;
  }
}

void sync_weights_to_host(Network *net) {
  cl_int err;

  // Sync hidden1 layer
  err = clEnqueueReadBuffer(queue, d_hidden1_weights, CL_TRUE, 0,
                            sizeof(float) * INPUT_SIZE * HIDDEN1_SIZE,
                            net->hidden1.weights, 0, NULL, NULL);
  check_error(err, "Reading hidden1 weights");

  err = clEnqueueReadBuffer(queue, d_hidden1_biases, CL_TRUE, 0,
                            sizeof(float) * HIDDEN1_SIZE, net->hidden1.biases,
                            0, NULL, NULL);
  check_error(err, "Reading hidden1 biases");

  err = clEnqueueReadBuffer(queue, d_hidden1_weight_momentum, CL_TRUE, 0,
                            sizeof(float) * INPUT_SIZE * HIDDEN1_SIZE,
                            net->hidden1.weight_momentum, 0, NULL, NULL);
  check_error(err, "Reading hidden1 weight momentum");

  err = clEnqueueReadBuffer(queue, d_hidden1_bias_momentum, CL_TRUE, 0,
                            sizeof(float) * HIDDEN1_SIZE,
                            net->hidden1.bias_momentum, 0, NULL, NULL);
  check_error(err, "Reading hidden1 bias momentum");

  // Sync hidden2 layer
  err = clEnqueueReadBuffer(queue, d_hidden2_weights, CL_TRUE, 0,
                            sizeof(float) * HIDDEN1_SIZE * HIDDEN2_SIZE,
                            net->hidden2.weights, 0, NULL, NULL);
  check_error(err, "Reading hidden2 weights");

  err = clEnqueueReadBuffer(queue, d_hidden2_biases, CL_TRUE, 0,
                            sizeof(float) * HIDDEN2_SIZE, net->hidden2.biases,
                            0, NULL, NULL);
  check_error(err, "Reading hidden2 biases");

  err = clEnqueueReadBuffer(queue, d_hidden2_weight_momentum, CL_TRUE, 0,
                            sizeof(float) * HIDDEN1_SIZE * HIDDEN2_SIZE,
                            net->hidden2.weight_momentum, 0, NULL, NULL);
  check_error(err, "Reading hidden2 weight momentum");

  err = clEnqueueReadBuffer(queue, d_hidden2_bias_momentum, CL_TRUE, 0,
                            sizeof(float) * HIDDEN2_SIZE,
                            net->hidden2.bias_momentum, 0, NULL, NULL);
  check_error(err, "Reading hidden2 bias momentum");

  // Sync output layer
  err = clEnqueueReadBuffer(queue, d_output_weights, CL_TRUE, 0,
                            sizeof(float) * HIDDEN2_SIZE * OUTPUT_SIZE,
                            net->output.weights, 0, NULL, NULL);
  check_error(err, "Reading output weights");

  err = clEnqueueReadBuffer(queue, d_output_biases, CL_TRUE, 0,
                            sizeof(float) * OUTPUT_SIZE, net->output.biases, 0,
                            NULL, NULL);
  check_error(err, "Reading output biases");

  err = clEnqueueReadBuffer(queue, d_output_weight_momentum, CL_TRUE, 0,
                            sizeof(float) * HIDDEN2_SIZE * OUTPUT_SIZE,
                            net->output.weight_momentum, 0, NULL, NULL);
  check_error(err, "Reading output weight momentum");

  err = clEnqueueReadBuffer(queue, d_output_bias_momentum, CL_TRUE, 0,
                            sizeof(float) * OUTPUT_SIZE,
                            net->output.bias_momentum, 0, NULL, NULL);
  check_error(err, "Reading output bias momentum");
}



void display_image(unsigned char *image) {
  for (int i = 0; i < IMAGE_SIZE; i++) {
    for (int j = 0; j < IMAGE_SIZE; j++) {
      unsigned char pixel = image[i * IMAGE_SIZE + j];
      if (pixel > 200)
        printf("@");
      else if (pixel > 150)
        printf("#");
      else if (pixel > 100)
        printf("*");
      else if (pixel > 50)
        printf(".");
      else
        printf(" ");
    }
    printf("\n");
  }
}


int predict(Network *net, float *input) {
    float final_output[OUTPUT_SIZE];
    
    // Perform forward pass through the network
    forward_opencl(input, NULL, NULL, final_output);
    
    // Find the index with highest probability
    int max_index = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (final_output[i] > final_output[max_index]) {
            max_index = i;
        }
    }
    
    return max_index;
}


int save_opencl_network(Network* net, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open file for writing");
        return -1;
    }

    // Header
    uint8_t magic = MAGIC_NUMBER;
    uint32_t version = FILE_VERSION;
    fwrite(&magic, sizeof(uint8_t), 1, fp);
    fwrite(&version, sizeof(uint32_t), 1, fp);

    // Network configuration
    float learning_rate = LEARNING_RATE;
    float momentum = MOMENTUM;
    int version_num = 1; // Default version
    fwrite(&learning_rate, sizeof(float), 1, fp);
    fwrite(&momentum, sizeof(float), 1, fp);
    fwrite(&version_num, sizeof(int), 1, fp);

    // Layer count (4 layers: input, hidden1, hidden2, output)
    uint8_t num_layers = 4;
    fwrite(&num_layers, sizeof(uint8_t), 1, fp);

    // Layer sizes
    uint32_t input_size = INPUT_SIZE;
    uint32_t hidden1_size = HIDDEN1_SIZE;
    uint32_t hidden2_size = HIDDEN2_SIZE;
    uint32_t output_size = OUTPUT_SIZE;
    
    fwrite(&input_size, sizeof(uint32_t), 1, fp);
    fwrite(&hidden1_size, sizeof(uint32_t), 1, fp);
    fwrite(&hidden2_size, sizeof(uint32_t), 1, fp);
    fwrite(&output_size, sizeof(uint32_t), 1, fp);

    // Activation functions (3 layers with activations: hidden1, hidden2, output)
    ActivationType hidden_act = ACT_RELU;
    ActivationType output_act = ACT_SOFTMAX;
    
    fwrite(&hidden_act, sizeof(ActivationType), 1, fp); // hidden1
    fwrite(&hidden_act, sizeof(ActivationType), 1, fp); // hidden2
    fwrite(&output_act, sizeof(ActivationType), 1, fp); // output

    // Weights and biases
    // First sync all weights from device to host
    sync_weights_to_host(net);

    // Hidden1 layer weights and biases
    size_t hidden1_weights_size = INPUT_SIZE * HIDDEN1_SIZE;
    if (fwrite(net->hidden1.weights, sizeof(float), hidden1_weights_size, fp) != hidden1_weights_size ||
        fwrite(net->hidden1.biases, sizeof(float), HIDDEN1_SIZE, fp) != HIDDEN1_SIZE) {
        fclose(fp);
        return -1;
    }

    // Hidden2 layer weights and biases
    size_t hidden2_weights_size = HIDDEN1_SIZE * HIDDEN2_SIZE;
    if (fwrite(net->hidden2.weights, sizeof(float), hidden2_weights_size, fp) != hidden2_weights_size ||
        fwrite(net->hidden2.biases, sizeof(float), HIDDEN2_SIZE, fp) != HIDDEN2_SIZE) {
        fclose(fp);
        return -1;
    }

    // Output layer weights and biases
    size_t output_weights_size = HIDDEN2_SIZE * OUTPUT_SIZE;
    if (fwrite(net->output.weights, sizeof(float), output_weights_size, fp) != output_weights_size ||
        fwrite(net->output.biases, sizeof(float), OUTPUT_SIZE, fp) != OUTPUT_SIZE) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}


int main() {
  Network net;
  InputData data = {0};
  float learning_rate = LEARNING_RATE, img[INPUT_SIZE];

  srand(time(NULL));

  init_layer(&net.hidden1, INPUT_SIZE, HIDDEN1_SIZE);
  init_layer(&net.hidden2, HIDDEN1_SIZE, HIDDEN2_SIZE);
  init_layer(&net.output, HIDDEN2_SIZE, OUTPUT_SIZE);

  init_opencl();
  create_opencl_buffers(&net);

  read_mnist_images(TRAIN_IMG_PATH, &data.images, &data.nImages);
  read_mnist_labels(TRAIN_LBL_PATH, &data.labels, &data.nImages);

  shuffle_data(data.images, data.labels, data.nImages);

  int train_size = (int)(data.nImages * TRAIN_SPLIT);
  int test_size = data.nImages - train_size;


  // Pick a random test image
  int idx = train_size + rand() % test_size;
  unsigned char *sample_img = &data.images[idx * INPUT_SIZE];
  int true_label = data.labels[idx];

  // Display the image
  printf("\nRandom Test Image (True Label: %d):\n", true_label);
  display_image(sample_img);



  double start, end;
  double epochstart, epochend;

  printf("Dataset: %d training samples, %d test samples\n", train_size,
         test_size);

  start = omp_get_wtime();

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    epochstart = omp_get_wtime();
    float total_loss = 0;

    for (int i = 0; i < train_size; i++) {
      for (int k = 0; k < INPUT_SIZE; k++)
        img[k] = data.images[i * INPUT_SIZE + k] / 255.0f;

      float *final_output =
          train_opencl(&net, img, data.labels[i], learning_rate);
      total_loss += -logf(final_output[data.labels[i]] + 1e-10f);
    }

    // Testing phase
    int correct = 0;
    for (int i = train_size; i < data.nImages; i++) {
      for (int k = 0; k < INPUT_SIZE; k++)
        img[k] = data.images[i * INPUT_SIZE + k] / 255.0f;

      if (predict_opencl(img) == data.labels[i])
        correct++;
    }

    epochend = omp_get_wtime();

    printf("Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f, Time: %.2f seconds\n",
           epoch + 1, (float)correct / test_size * 100, total_loss / train_size,
           epochend-epochstart);
  }

  end = omp_get_wtime();


  for (int k = 0; k < INPUT_SIZE; k++)
    img[k] = sample_img[k] / 255.0f;

  int predicted = predict(&net, img);
  printf("Predicted Label: %d\n", predicted);


  sync_weights_to_host(&net);
  
  printf("Total time needed: %.2f seconds\n", end-start);



save_opencl_network(&net, "ocl2.nn");

   // Free network memory
   cleanup_opencl();


  free(net.hidden2.weights);
  free(net.hidden2.biases);
  free(net.hidden2.weight_momentum);
  free(net.hidden2.bias_momentum);

  free(net.hidden1.weights);
  free(net.hidden1.biases);
  free(net.hidden1.weight_momentum);
  free(net.hidden1.bias_momentum);
  free(net.output.weights);
  free(net.output.biases);
  free(net.output.weight_momentum);
  free(net.output.bias_momentum);
  free(data.images);
  free(data.labels);

  return 0;
}
