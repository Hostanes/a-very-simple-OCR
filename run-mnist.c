#include "lib/nnlib-ocl.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <CL/cl.h>

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"
#define TEST_IMG_PATH "data/t10k-images.idx3-ubyte"
#define TEST_LBL_PATH "data/t10k-labels.idx1-ubyte"

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.05f
#define MOMENTUM 0.9f
#define EPOCHS 5
#define IMAGE_SIZE 28
#define PRINT_INTERVAL 100

typedef struct {
  unsigned char *images, *labels;
  int nImages;
} Dataset;

// Function declarations
void load_mnist(const char *image_path, const char *label_path,
                Dataset *dataset);
void shuffle_data(Dataset *dataset);
void normalize_images(unsigned char *input, float *output, int count);
void one_hot_encode(unsigned char label, float *output);
float calculate_accuracy(NeuralNetwork_t *net, cl_command_queue queue,
                         float *test_images, unsigned char *test_labels,
                         int test_count);

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

int main() {
  srand(time(NULL));

  // Initialize OpenCL
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_int err;

  err = clGetPlatformIDs(1, &platform, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error: Failed to find OpenCL platform.\n");
    return EXIT_FAILURE;
  }

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Warning: No GPU found, trying CPU...\n");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
      fprintf(stderr, "Error: Failed to find any OpenCL device.\n");
      return EXIT_FAILURE;
    }
  }

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error: Failed to create OpenCL context.\n");
    return EXIT_FAILURE;
  }

  queue = clCreateCommandQueue(context, device, 0, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error: Failed to create command queue.\n");
    clReleaseContext(context);
    return EXIT_FAILURE;
  }

  // Load MNIST data
  Dataset train_data, test_data;
  load_mnist(TRAIN_IMG_PATH, TRAIN_LBL_PATH, &train_data);
  load_mnist(TEST_IMG_PATH, TEST_LBL_PATH, &test_data);

  // Normalize and prepare data
  float *train_images =
      (float *)malloc(train_data.nImages * INPUT_SIZE * sizeof(float));
  float *test_images =
      (float *)malloc(test_data.nImages * INPUT_SIZE * sizeof(float));
  normalize_images(train_data.images, train_images, train_data.nImages);
  normalize_images(test_data.images, test_images, test_data.nImages);

  // Shuffle training data
  shuffle_data(&train_data);

  // Create neural network
  int layer_sizes[] = {INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};
  NeuralNetwork_t net = create_NeuralNetwork(context, layer_sizes, 3);

  // Load and build OpenCL program
  char *kernel_source = load_kernel_source("lib/kernels.cl");
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&kernel_source, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error: Failed to create program from source.\n");
    return EXIT_FAILURE;
  }

  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    char *log = (char *)malloc(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log,
                          NULL);
    fprintf(stderr, "Build error:\n%s\n", log);
    free(log);
    return EXIT_FAILURE;
  }

  // Create kernels
  net.forward_relu_kernel = clCreateKernel(program, "forward_With_Relu", &err);
  net.forward_softmax_kernel =
      clCreateKernel(program, "forward_With_Softmax", &err);
  net.backward_relu_kernel =
      clCreateKernel(program, "backward_With_Relu", &err);
  net.backward_softmax_kernel =
      clCreateKernel(program, "backward_With_Softmax", &err);
  net.program = program;
  net.queue = queue;

  // Initialize weights randomly
  for (int i = 0; i < net.num_layers; i++) {
    Layer_t *layer = &net.layers[i];
    float *weights =
        (float *)malloc(layer->input_size * layer->output_size * sizeof(float));
    float *biases = (float *)malloc(layer->output_size * sizeof(float));

    // Replace the random initialization with:
    for (int j = 0; j < layer->input_size * layer->output_size; j++) {
      weights[j] =
          ((float)rand() / RAND_MAX - 0.5f) * sqrtf(2.0f / layer->input_size);
    }
    for (int j = 0; j < layer->output_size; j++) {
      biases[j] = 0.01f; // Small positive bias
    }

    clEnqueueWriteBuffer(queue, layer->weights, CL_TRUE, 0,
                         layer->input_size * layer->output_size * sizeof(float),
                         weights, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, layer->biases, CL_TRUE, 0,
                         layer->output_size * sizeof(float), biases, 0, NULL,
                         NULL);
    free(weights);
    free(biases);
  }

  // Training loop
  printf("Starting training...\n");
  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    double start_time = omp_get_wtime();
    float epoch_loss = 0.0f;
    int correct = 0;

    for (int i = 0; i < train_data.nImages; i++) {
      float target[OUTPUT_SIZE];
      one_hot_encode(train_data.labels[i], target);

      // Forward pass
      forward(&net, queue, train_images + i * INPUT_SIZE, INPUT_SIZE, NULL, 0);

      // Backward pass
      backward(&net, queue, target, OUTPUT_SIZE, LEARNING_RATE, MOMENTUM);

      // Calculate accuracy on this example
      float output[OUTPUT_SIZE];
      forward(&net, queue, train_images + i * INPUT_SIZE, INPUT_SIZE, output,
              OUTPUT_SIZE);

      int pred = 0;
      for (int j = 1; j < OUTPUT_SIZE; j++) {
        if (output[j] > output[pred])
          pred = j;
      }
      if (pred == train_data.labels[i])
        correct++;

      if ((i + 1) % PRINT_INTERVAL == 0) {
        float accuracy = calculate_accuracy(&net, queue, test_images,
                                            test_data.labels, 1000);
        printf("Epoch %d, Example %d/%d - Train Acc (batch): %.2f%%, Test Acc: "
               "%.2f%%\n",
               epoch + 1, i + 1, train_data.nImages,
               (float)correct / (i + 1) * 100, accuracy * 100);
      }
    }

    // Calculate epoch statistics
    float train_acc = (float)correct / train_data.nImages * 100;
    float test_acc = calculate_accuracy(&net, queue, test_images,
                                        test_data.labels, test_data.nImages);
    double end_time = omp_get_wtime();
    printf("Epoch %d complete - Time: %.2f seconds, Train Acc: %.2f%%, Test "
           "Acc: %.2f%%\n",
           epoch + 1, end_time - start_time, train_acc, test_acc * 100);
  }

  // Cleanup
  free(train_images);
  free(test_images);
  free(train_data.images);
  free(train_data.labels);
  free(test_data.images);
  free(test_data.labels);
  free_NeuralNetwork(&net);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(kernel_source);

  return EXIT_SUCCESS;
}

void load_mnist(const char *image_path, const char *label_path,
                Dataset *dataset) {
  FILE *file = fopen(image_path, "rb");
  if (!file) {
    perror("Error opening image file");
    exit(EXIT_FAILURE);
  }

  int magic_number, n_images, rows, cols;
  fread(&magic_number, sizeof(int), 1, file);
  fread(&n_images, sizeof(int), 1, file);
  fread(&rows, sizeof(int), 1, file);
  fread(&cols, sizeof(int), 1, file);

  // Convert from big endian to host byte order
  magic_number = __builtin_bswap32(magic_number);
  n_images = __builtin_bswap32(n_images);
  rows = __builtin_bswap32(rows);
  cols = __builtin_bswap32(cols);

  dataset->images = (unsigned char *)malloc(n_images * rows * cols);
  fread(dataset->images, sizeof(unsigned char), n_images * rows * cols, file);
  fclose(file);

  file = fopen(label_path, "rb");
  if (!file) {
    perror("Error opening label file");
    exit(EXIT_FAILURE);
  }

  fread(&magic_number, sizeof(int), 1, file);
  fread(&n_images, sizeof(int), 1, file);
  magic_number = __builtin_bswap32(magic_number);
  n_images = __builtin_bswap32(n_images);

  dataset->labels = (unsigned char *)malloc(n_images);
  fread(dataset->labels, sizeof(unsigned char), n_images, file);
  fclose(file);

  dataset->nImages = n_images;
}

void shuffle_data(Dataset *dataset) {
  for (int i = dataset->nImages - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    // Swap images
    for (int k = 0; k < INPUT_SIZE; k++) {
      unsigned char temp = dataset->images[i * INPUT_SIZE + k];
      dataset->images[i * INPUT_SIZE + k] = dataset->images[j * INPUT_SIZE + k];
      dataset->images[j * INPUT_SIZE + k] = temp;
    }
    // Swap labels
    unsigned char temp = dataset->labels[i];
    dataset->labels[i] = dataset->labels[j];
    dataset->labels[j] = temp;
  }
}

void normalize_images(unsigned char *input, float *output, int count) {
  for (int i = 0; i < count * INPUT_SIZE; i++) {
    output[i] = input[i] / 255.0f;
  }
}

void one_hot_encode(unsigned char label, float *output) {
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    output[i] = (i == label) ? 1.0f : 0.0f;
  }
}

float calculate_accuracy(NeuralNetwork_t *net, cl_command_queue queue,
                         float *test_images, unsigned char *test_labels,
                         int test_count) {
  int correct = 0;
  float *output = (float *)malloc(OUTPUT_SIZE * sizeof(float));

  for (int i = 0; i < test_count; i++) {
    forward(net, queue, test_images + i * INPUT_SIZE, INPUT_SIZE, output,
            OUTPUT_SIZE);

    int pred = 0;
    for (int j = 1; j < OUTPUT_SIZE; j++) {
      if (output[j] > output[pred])
        pred = j;
    }

    if (pred == test_labels[i])
      correct++;
  }

  free(output);
  return (float)correct / test_count;
}
