#include "lib/nnlib-ocl.h"
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define IMAGE_SIZE 28
#define EPSILON 0.0001f

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

typedef struct {
  unsigned char *images;
  unsigned char *labels;
  int nImages;
} Dataset;

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

// Function to convert the image data to float and normalize it.
void preprocess_images(unsigned char *images, int num_images, int image_size,
                       float *normalized_images) {
  for (int i = 0; i < num_images; i++) {
    for (int j = 0; j < image_size; j++) {
      normalized_images[i * image_size + j] =
          (float)images[i * image_size + j] / 255.0f;
    }
  }
}

// Function to convert the labels to one-hot encoded vectors.
void one_hot_encode_labels(unsigned char *labels, int num_labels,
                           int num_classes, float *one_hot_labels) {
  for (int i = 0; i < num_labels; i++) {
    for (int j = 0; j < num_classes; j++) {
      one_hot_labels[i * num_classes + j] = 0.0f;
    }
    one_hot_labels[i * num_classes + labels[i]] = 1.0f;
  }
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

  queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error: Failed to create command queue.\n");
    clReleaseContext(context);
    return 1;
  }

  // Load MNIST dataset
  Dataset training_data;
  load_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
             &training_data);
  Dataset test_data;
  load_mnist("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
             &test_data);

  // Preprocess images and labels
  const int image_size = 28 * 28; // 784
  const int num_classes = 10;
  float *train_images_normalized =
      (float *)malloc(training_data.nImages * image_size * sizeof(float));
  float *train_labels_one_hot =
      (float *)malloc(training_data.nImages * num_classes * sizeof(float));
  float *test_images_normalized =
      (float *)malloc(test_data.nImages * image_size * sizeof(float));
  float *test_labels_one_hot =
      (float *)malloc(test_data.nImages * num_classes * sizeof(float));

  preprocess_images(training_data.images, training_data.nImages, image_size,
                    train_images_normalized);
  one_hot_encode_labels(training_data.labels, training_data.nImages,
                        num_classes, train_labels_one_hot);
  preprocess_images(test_data.images, test_data.nImages, image_size,
                    test_images_normalized);
  one_hot_encode_labels(test_data.labels, test_data.nImages, num_classes,
                        test_labels_one_hot);

  srand(time(NULL)); // Seed the random number generator
  int random_index = rand() % training_data.nImages;
  printf("Displaying a random training image (index %d):\n", random_index);
  display_image(&training_data.images[random_index * image_size]);
  printf("Label: %d\n", training_data.labels[random_index]);

  // Define network architecture: 784 -> 512 -> 256 -> 10
  int layer_sizes[] = {784, 512, 256, 10};
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

  // Set the learning rate and momentum.  These are common values, but you
  // should experiment with them.
  float learning_rate = 0.001f;
  float momentum = 0.5f;
  int batch_size = 64; 
  int num_epochs = 1;  

  // Training loop
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    printf("Epoch %d/%d\n", epoch + 1, num_epochs);
    for (int i = 0; i < training_data.nImages / batch_size; i++) {
      // Get the current batch of data.
      float *input_batch =
          train_images_normalized + i * batch_size * image_size;
      float *target_batch = train_labels_one_hot + i * batch_size * num_classes;

      // Train the network on the batch.
      for (int j = 0; j < batch_size; j++) {
        train(&net, queue, input_batch + j * image_size, image_size,
              target_batch + j * num_classes, num_classes, learning_rate,
              momentum);
      }
    }
  }

  // Test the network on the test data.
  int correct_predictions = 0;
  for (int i = 0; i < test_data.nImages; i++) {
    float input_image[image_size];
    for (int j = 0; j < image_size; j++) {
      input_image[j] = test_images_normalized[i * image_size + j];
    }
    float output[num_classes];
    forward(&net, queue, input_image, image_size, output, num_classes);

    // Get the predicted label.
    int predicted_label = 0;
    float max_output = output[0];
    for (int j = 1; j < num_classes; j++) {
      if (output[j] > max_output) {
        max_output = output[j];
        predicted_label = j;
      }
    }

    // Get the actual label.
    int actual_label = test_data.labels[i];

    // Check if the prediction was correct.
    if (predicted_label == actual_label) {
      correct_predictions++;
    }
  }

  // Print the accuracy.
  float accuracy = (float)correct_predictions / test_data.nImages;
  printf("Accuracy: %.2f%%\n", accuracy * 100.0f);

  // Cleanup
  free(training_data.images);
  free(training_data.labels);
  free(train_images_normalized);
  free(train_labels_one_hot);
  free(test_data.images);
  free(test_data.labels);
  free(test_images_normalized);
  free(test_labels_one_hot);
  free_NeuralNetwork(&net);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return 0;
}
