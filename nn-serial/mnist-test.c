#include "lib/nnlib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Constants
#define IMAGE_SIZE 28
#define INPUT_SIZE (IMAGE_SIZE * IMAGE_SIZE)
#define NUM_CLASSES 10

// Data structure to hold MNIST data as flat arrays
typedef struct {
  float *images; // Flattened images (num_images × INPUT_SIZE)
  int *labels;   // Labels (num_images)
  int num_images;
} MNISTData;

// Function to read MNIST images as flat 1D array
void read_mnist_images(const char *filename, float **images, int *num_images) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    perror("Error opening image file");
    exit(EXIT_FAILURE);
  }

  // Read header information
  int magic_number, rows, cols;
  fread(&magic_number, sizeof(int), 1, file);
  fread(num_images, sizeof(int), 1, file);
  fread(&rows, sizeof(int), 1, file);
  fread(&cols, sizeof(int), 1, file);

  // Convert from big-endian to native byte order
  magic_number = __builtin_bswap32(magic_number);
  *num_images = __builtin_bswap32(*num_images);
  rows = __builtin_bswap32(rows);
  cols = __builtin_bswap32(cols);

  // Validate dimensions
  if (rows != IMAGE_SIZE || cols != IMAGE_SIZE) {
    fprintf(stderr,
            "Error: Image dimensions mismatch (expected %dx%d, got %dx%d)\n",
            IMAGE_SIZE, IMAGE_SIZE, rows, cols);
    fclose(file);
    exit(EXIT_FAILURE);
  }

  // Allocate memory for images (already flattened)
  *images = (float *)malloc(*num_images * INPUT_SIZE * sizeof(float));
  if (!*images) {
    perror("Error allocating memory for images");
    fclose(file);
    exit(EXIT_FAILURE);
  }

  // Read image data and normalize to [0,1]
  unsigned char *temp_buffer =
      (unsigned char *)malloc(*num_images * INPUT_SIZE);
  if (!temp_buffer) {
    perror("Error allocating temporary buffer");
    fclose(file);
    exit(EXIT_FAILURE);
  }

  fread(temp_buffer, sizeof(unsigned char), *num_images * INPUT_SIZE, file);
  fclose(file);

  // Convert to float and normalize
  for (int i = 0; i < *num_images * INPUT_SIZE; i++) {
    (*images)[i] = temp_buffer[i] / 255.0f;
  }

  free(temp_buffer);
}

// Function to read MNIST labels as flat 1D array
void read_mnist_labels(const char *filename, int **labels, int *num_labels) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    perror("Error opening label file");
    exit(EXIT_FAILURE);
  }

  // Read header information
  int magic_number;
  fread(&magic_number, sizeof(int), 1, file);
  fread(num_labels, sizeof(int), 1, file);

  // Convert from big-endian to native byte order
  magic_number = __builtin_bswap32(magic_number);
  *num_labels = __builtin_bswap32(*num_labels);

  // Allocate memory for labels
  *labels = (int *)malloc(*num_labels * sizeof(int));
  if (!*labels) {
    perror("Error allocating memory for labels");
    fclose(file);
    exit(EXIT_FAILURE);
  }

  // Read label data
  unsigned char *temp_buffer = (unsigned char *)malloc(*num_labels);
  if (!temp_buffer) {
    perror("Error allocating temporary buffer");
    fclose(file);
    exit(EXIT_FAILURE);
  }

  fread(temp_buffer, sizeof(unsigned char), *num_labels, file);
  fclose(file);

  // Convert to int
  for (int i = 0; i < *num_labels; i++) {
    (*labels)[i] = (int)temp_buffer[i];
  }

  free(temp_buffer);
}

// Shuffle function for flat arrays
void shuffle_data(float *images, int *labels, int num_images) {
  if (num_images <= 1)
    return;

  srand(time(NULL));
  for (int i = num_images - 1; i > 0; i--) {
    int j = rand() % (i + 1);

    // Swap images (all INPUT_SIZE pixels at once)
    float temp_image[INPUT_SIZE];
    memcpy(temp_image, &images[i * INPUT_SIZE], INPUT_SIZE * sizeof(float));
    memcpy(&images[i * INPUT_SIZE], &images[j * INPUT_SIZE],
           INPUT_SIZE * sizeof(float));
    memcpy(&images[j * INPUT_SIZE], temp_image, INPUT_SIZE * sizeof(float));

    // Swap labels
    int temp_label = labels[i];
    labels[i] = labels[j];
    labels[j] = temp_label;
  }
}

int main(int argc, char *argv[]) {
  const char *train_images_file = "data/train-images.idx3-ubyte";
  const char *train_labels_file = "data/train-labels.idx1-ubyte";
  const char *test_images_file = "data/t10k-images.idx3-ubyte";
  const char *test_labels_file = "data/t10k-labels.idx1-ubyte";

  // Load training data
  MNISTData train_data;
  read_mnist_images(train_images_file, &train_data.images,
                    &train_data.num_images);
  read_mnist_labels(train_labels_file, &train_data.labels,
                    &train_data.num_images);

  // Load test data
  MNISTData test_data;
  read_mnist_images(test_images_file, &test_data.images, &test_data.num_images);
  read_mnist_labels(test_labels_file, &test_data.labels, &test_data.num_images);

  printf("Loaded %d training images and %d test images\n",
         train_data.num_images, test_data.num_images);

  // Define network architecture
  int layer_sizes[] = {INPUT_SIZE, 512, 256, NUM_CLASSES};
  int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);

  // Initialize weights and biases as flat arrays
  int total_weights = 0;
  int total_biases = 0;
  for (int i = 0; i < num_layers - 1; i++) {
    total_weights += layer_sizes[i] * layer_sizes[i + 1];
    total_biases += layer_sizes[i + 1];
  }

  float *weights = (float *)malloc(total_weights * sizeof(float));
  float *biases = (float *)malloc(total_biases * sizeof(float));

  // Initialize weights randomly
  srand(time(NULL));
  for (int i = 0; i < total_weights; i++) {
    weights[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
  }
  memset(biases, 0, total_biases * sizeof(float));

  // Training parameters
  int batch_size = 16;
  int num_epochs = 1;
  float learning_rate = 0.01f;

  // Training loop
  printf("\n--- Training ---\n");
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    printf("Epoch %d/%d\n", epoch + 1, num_epochs);

    // Shuffle training data
    shuffle_data(train_data.images, train_data.labels, train_data.num_images);

    for (int i = 0; i < train_data.num_images; i += batch_size) {
      int current_batch_size = (i + batch_size <= train_data.num_images)
                                   ? batch_size
                                   : (train_data.num_images - i);

      // Get current batch (already flat)
      float *batch_images = &train_data.images[i * INPUT_SIZE];
      int *batch_labels = &train_data.labels[i];

      // Allocate neuron values buffer (flat array)
      int max_neurons = 0;
      for (int l = 0; l < num_layers; l++) {
        if (layer_sizes[l] > max_neurons)
          max_neurons = layer_sizes[l];
      }
      float *neuron_values =
          (float *)malloc(current_batch_size * 2 * max_neurons * sizeof(float));

      // Forward pass
      forward_Pass(batch_images, weights, biases, neuron_values,
                   current_batch_size, layer_sizes, num_layers);

      // Calculate output errors (flat array)
      float *output_errors =
          (float *)malloc(current_batch_size * NUM_CLASSES * sizeof(float));
      for (int s = 0; s < current_batch_size; s++) {
        for (int c = 0; c < NUM_CLASSES; c++) {
          float output = neuron_values[s * 2 * NUM_CLASSES + 2 * c + 1];
          output_errors[s * NUM_CLASSES + c] =
              (c == batch_labels[s]) ? (output - 1.0f) : output;
        }
      }

      // Backward pass
      float *gradient_weights = (float *)malloc(total_weights * sizeof(float));
      float *gradient_biases = (float *)malloc(total_biases * sizeof(float));

      backward_Pass(batch_images, weights, neuron_values, gradient_weights,
                    gradient_biases, output_errors, current_batch_size,
                    layer_sizes, num_layers);

      // Update weights and biases
      for (int w = 0; w < total_weights; w++) {
        weights[w] -= learning_rate * gradient_weights[w];
      }
      for (int b = 0; b < total_biases; b++) {
        biases[b] -= learning_rate * gradient_biases[b];
      }

      // Free temporary buffers
      free(neuron_values);
      free(output_errors);
      free(gradient_weights);
      free(gradient_biases);
    }
  }

  // Testing
  printf("\n--- Testing ---\n");
  int correct = 0;
  for (int i = 0; i < test_data.num_images; i += batch_size) {
    int current_batch_size = (i + batch_size <= test_data.num_images)
                                 ? batch_size
                                 : (test_data.num_images - i);

    float *batch_images = &test_data.images[i * INPUT_SIZE];
    int *batch_labels = &test_data.labels[i];

    // Allocate neuron values buffer
    int max_neurons = 0;
    for (int l = 0; l < num_layers; l++) {
      if (layer_sizes[l] > max_neurons)
        max_neurons = layer_sizes[l];
    }
    float *neuron_values =
        (float *)malloc(current_batch_size * 2 * max_neurons * sizeof(float));

    // Forward pass
    forward_Pass(batch_images, weights, biases, neuron_values,
                 current_batch_size, layer_sizes, num_layers);

    // Check predictions
    for (int s = 0; s < current_batch_size; s++) {
      int predicted = 0;
      float max_prob = -1.0f;
      for (int c = 0; c < NUM_CLASSES; c++) {
        float prob = neuron_values[s * 2 * NUM_CLASSES + 2 * c + 1];
        if (prob > max_prob) {
          max_prob = prob;
          predicted = c;
        }
      }
      if (predicted == batch_labels[s]) {
        correct++;
      }
    }

    free(neuron_values);
  }

  printf("Test Accuracy: %.2f%%\n",
         (float)correct / test_data.num_images * 100.0f);

  // Clean up
  free(train_data.images);
  free(train_data.labels);
  free(test_data.images);
  free(test_data.labels);
  free(weights);
  free(biases);

  return 0;
}
