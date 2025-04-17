
#include "lib/nnlib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"
#define INPUT_SIZE 784
#define IMAGE_SIZE 28

typedef struct {
  unsigned char *images;
  unsigned char *labels;
  int nImages;
} MNISTData;

void read_mnist_images(const char *filename, unsigned char **images,
                       int *nImages);
void read_mnist_labels(const char *filename, unsigned char **labels,
                       int *nLabels);
void display_image(unsigned char *image);
void normalize_image(unsigned char *input, float *output);

int main() {
  // Initialize random seed
  srand(time(NULL));

  // Load both models
  NeuralNetwork_t *parallel_model = load_Network("parallel.nn");
  NeuralNetwork_t *serial_model = load_Network("serial.nn");

  if (!parallel_model || !serial_model) {
    fprintf(stderr, "Failed to load one or both models\n");
    return 1;
  }

  // Load MNIST test data
  MNISTData test_data = {0};
  read_mnist_images(TRAIN_IMG_PATH, &test_data.images, &test_data.nImages);
  read_mnist_labels(TRAIN_LBL_PATH, &test_data.labels, &test_data.nImages);

  // Select a random test image
  int random_index = rand() % test_data.nImages;
  unsigned char *image = &test_data.images[random_index * INPUT_SIZE];
  int true_label = test_data.labels[random_index];

  // Display the selected image
  printf("\nSelected Test Image (True Label: %d):\n", true_label);
  display_image(image);

  // Prepare input for network
  float normalized_input[INPUT_SIZE];
  normalize_image(image, normalized_input);

  // Get predictions from both models
  float *parallel_output = forward_pass(parallel_model, normalized_input);
  float *serial_output = forward_pass(serial_model, normalized_input);

  // Print comparison
  printf("\nModel Comparison:\n");
  printf("%-10s %-15s %-15s\n", "Class", "Parallel Output", "Serial Output");
  printf("----------------------------------------\n");

  for (int i = 0; i < 10; i++) {
    printf("%-10d %-15.6f %-15.6f\n", i, parallel_output[i], serial_output[i]);
  }

  // Get predicted classes
  int parallel_pred = predict(parallel_model, normalized_input);
  int serial_pred = predict(serial_model, normalized_input);

  printf("\nPredictions:\n");
  printf("Parallel Model: %d (confidence: %.2f%%)\n", parallel_pred,
         parallel_output[parallel_pred] * 100);
  printf("Serial Model:   %d (confidence: %.2f%%)\n", serial_pred,
         serial_output[serial_pred] * 100);
  printf("True Label:     %d\n", true_label);

  // Clean up
  free_network(parallel_model);
  free_network(serial_model);
  free(test_data.images);
  free(test_data.labels);

  return 0;
}

void read_mnist_images(const char *filename, unsigned char **images,
                       int *nImages) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    perror("Failed to open images file");
    exit(1);
  }

  int magic, rows, cols;
  fread(&magic, sizeof(int), 1, file);
  fread(nImages, sizeof(int), 1, file);

  // MNIST data is in big-endian format
  magic = __builtin_bswap32(magic);
  *nImages = __builtin_bswap32(*nImages);

  fread(&rows, sizeof(int), 1, file);
  fread(&cols, sizeof(int), 1, file);
  rows = __builtin_bswap32(rows);
  cols = __builtin_bswap32(cols);

  *images = malloc((*nImages) * rows * cols);
  fread(*images, sizeof(unsigned char), (*nImages) * rows * cols, file);
  fclose(file);
}

void read_mnist_labels(const char *filename, unsigned char **labels,
                       int *nLabels) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    perror("Failed to open labels file");
    exit(1);
  }

  int magic;
  fread(&magic, sizeof(int), 1, file);
  fread(nLabels, sizeof(int), 1, file);

  magic = __builtin_bswap32(magic);
  *nLabels = __builtin_bswap32(*nLabels);

  *labels = malloc(*nLabels);
  fread(*labels, sizeof(unsigned char), *nLabels, file);
  fclose(file);
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

void normalize_image(unsigned char *input, float *output) {
  for (int i = 0; i < INPUT_SIZE; i++) {
    output[i] = input[i] / 255.0f;
  }
}
