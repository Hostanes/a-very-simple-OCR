#include "lib/nnlib.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.0005f
#define MOMENTUM 0.9f
#define EPOCHS 5
#define BATCH_SIZE 64
#define IMAGE_SIZE 28
#define TRAIN_SPLIT 0.8
#define PRINT_INTERVAL 1000

typedef struct {
  unsigned char *images, *labels;
  int nImages;
} InputData_t;

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

void read_mnist_images(const char *filename, unsigned char **images,
                       int *nImages) {
  FILE *file = fopen(filename, "rb");
  if (!file)
    exit(1);

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
  if (!file)
    exit(1);

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

void normalize_images(unsigned char *input, float *output, int count) {
  for (int i = 0; i < INPUT_SIZE * count; i++) {
    output[i] = input[i] / 255.0f;
  }
}

void print_usage(const char *program_name) {
  printf("Usage: %s <output_filename.nn>\n", program_name);
  printf("Example: %s my_model.nn\n", program_name);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    print_usage(argv[0]);
    return 1;
  }

  const char *output_filename = argv[1];

  InputData_t data = {0};
  double start_time, end_time;
  double time_taken;

  int layer_Sizes[4] = {784, 512, 256, 10};
  ActivationFunc activations[3] = {relu, relu, softmax_placeholder};
  ActivationDerivative derivatives[3] = {relu_derivative, relu_derivative,
                                         NULL};

  srand(time(NULL));
  NeuralNetwork_t *net = create_network(layer_Sizes, 4, activations,
                                        derivatives, LEARNING_RATE, MOMENTUM);

  read_mnist_images(TRAIN_IMG_PATH, &data.images, &data.nImages);
  read_mnist_labels(TRAIN_LBL_PATH, &data.labels, &data.nImages);

  shuffle_data(data.images, data.labels, data.nImages);

  int train_size = (int)(data.nImages * TRAIN_SPLIT);
  int test_size = data.nImages - train_size;

  float img[INPUT_SIZE];
  float target[OUTPUT_SIZE];

  // Pick a random test image
  int idx = train_size + rand() % test_size;
  unsigned char *sample_img = &data.images[idx * INPUT_SIZE];
  int true_label = data.labels[idx];

  // Display the image
  printf("\nRandom Test Image (True Label: %d):\n", true_label);
  display_image(sample_img);

  printf("Starting training...\n");
  printf("Model will be saved to: %s\n", output_filename);

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    start_time = omp_get_wtime();
    float total_loss = 0;

    for (int i = 0; i < train_size; i++) {
      normalize_images(&data.images[i * INPUT_SIZE], img, 1);

      memset(target, 0, sizeof(target));
      target[data.labels[i]] = 1.0f;

      train(net, img, target);
      float *output = net->layers[net->num_layers - 1].output;
      total_loss += -logf(output[data.labels[i]] + 1e-10f);
    }

    int correct = 0;
    for (int i = train_size; i < data.nImages; i++) {
      normalize_images(&data.images[i * INPUT_SIZE], img, 1);
      if (predict(net, img) == data.labels[i])
        correct++;
    }

    end_time = omp_get_wtime();
    time_taken = end_time - start_time;

    printf("Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f, Time: %.2f seconds\n",
           epoch + 1, (float)correct / test_size * 100, total_loss / train_size,
           time_taken);
  }

  for (int k = 0; k < INPUT_SIZE; k++)
    img[k] = sample_img[k] / 255.0f;

  int predicted = predict(net, img);
  printf("Predicted Label: %d\n", predicted);

  if (save_Network(net, output_filename) != 0) {
    fprintf(stderr, "Error: Failed to save model to %s\n", output_filename);
    free_network(net);
    free(data.images);
    free(data.labels);
    return 1;
  }

  printf("Model successfully saved to %s\n", output_filename);

  free_network(net);
  free(data.images);
  free(data.labels);

  return 0;
}
