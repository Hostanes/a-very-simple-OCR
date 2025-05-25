
#include "batched/nnlib.h"
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
#define BATCH_SIZE 128
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
  srand(time(NULL));

  int layer_Sizes[4] = {INPUT_SIZE, 512, HIDDEN_SIZE, OUTPUT_SIZE};
  ActivationFunc activations[3] = {relu, relu, softmax_placeholder};
  ActivationDerivative derivatives[3] = {relu_derivative, relu_derivative,
                                         NULL};

  NeuralNetwork_t *net =
      create_network(layer_Sizes, 4, activations, derivatives, LEARNING_RATE,
                     MOMENTUM, BATCH_SIZE);

  read_mnist_images(TRAIN_IMG_PATH, &data.images, &data.nImages);
  read_mnist_labels(TRAIN_LBL_PATH, &data.labels, &data.nImages);
  shuffle_data(data.images, data.labels, data.nImages);

  int train_size = (int)(data.nImages * TRAIN_SPLIT);
  int test_size = data.nImages - train_size;

  float **batch_inputs = malloc(BATCH_SIZE * sizeof(float *));
  float **batch_targets = malloc(BATCH_SIZE * sizeof(float *));
  for (int i = 0; i < BATCH_SIZE; i++) {
    batch_inputs[i] = malloc(INPUT_SIZE * sizeof(float));
    batch_targets[i] = malloc(OUTPUT_SIZE * sizeof(float));
  }

  printf("Starting training...\n");
  printf("Model will be saved to: %s\n", output_filename);

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    double start_time = omp_get_wtime();
    float total_loss = 0;

    for (int i = 0; i < train_size; i += BATCH_SIZE) {
      int current_batch =
          (i + BATCH_SIZE <= train_size) ? BATCH_SIZE : (train_size - i);

      for (int j = 0; j < current_batch; j++) {
        normalize_images(&data.images[(i + j) * INPUT_SIZE], batch_inputs[j],
                         1);
        memset(batch_targets[j], 0, OUTPUT_SIZE * sizeof(float));
        batch_targets[j][data.labels[i + j]] = 1.0f;
      }

      resize_network_batch(net, current_batch);
      train_batch(net, batch_inputs, batch_targets, current_batch);

      float **outputs = forward_pass_batch(net, batch_inputs, current_batch);
      total_loss +=
          calculate_batch_loss(net, outputs, batch_targets, current_batch);
    }

    int correct = 0;
    for (int i = train_size; i < data.nImages; i += BATCH_SIZE) {
      int current_batch =
          (i + BATCH_SIZE <= data.nImages) ? BATCH_SIZE : (data.nImages - i);

      for (int j = 0; j < current_batch; j++) {
        normalize_images(&data.images[(i + j) * INPUT_SIZE], batch_inputs[j],
                         1);
      }

      resize_network_batch(net, current_batch);
      int *predictions = predict_batch(net, batch_inputs, current_batch);

      for (int j = 0; j < current_batch; j++) {
        if (predictions[j] == data.labels[i + j])
          correct++;
      }

      free(predictions);
    }

    double end_time = omp_get_wtime();
    printf("Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f, Time: %.2f seconds\n",
           epoch + 1, (float)correct / test_size * 100, total_loss / train_size,
           end_time - start_time);
  }

  // Random test image prediction
  int idx = train_size + rand() % test_size;
  printf("\nRandom Test Image (True Label: %d):\n", data.labels[idx]);
  display_image(&data.images[idx * INPUT_SIZE]);

  float *img = batch_inputs[0];
  normalize_images(&data.images[idx * INPUT_SIZE], img, 1);
  int predicted = predict_batch(net, &img, 1)[0];
  printf("Predicted Label: %d\n", predicted);


  // Free
  for (int i = 0; i < BATCH_SIZE; i++) {
    free(batch_inputs[i]);
    free(batch_targets[i]);
  }
  free(batch_inputs);
  free(batch_targets);
  free_network(net);
  free(data.images);
  free(data.labels);
  return 0;
}
