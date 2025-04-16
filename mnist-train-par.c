
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
#define LEARNING_RATE 0.001f
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

int main() {
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

  int total_weights = 0;
  {
    for (int l = 1; l < net->num_layers; l++) {
      Layer_t *layer = &net->layers[l];
      total_weights += layer->input_size * layer->output_size;
    }
  }

  printf("starting training\n");

  int batch_size = 8;
  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    start_time = omp_get_wtime();
    float total_loss = 0;

    // Shuffle data at start of each epoch
    shuffle_data(data.images, data.labels, data.nImages);

// Parallel region with reduction
#pragma omp parallel reduction(+ : total_loss)
    {
      // Thread-local gradient buffers
      float **thread_gradients = malloc(net->num_layers * sizeof(float *));
      float **thread_bias_gradients = malloc(net->num_layers * sizeof(float *));

      for (int l = 0; l < net->num_layers; l++) {
        int weights_size =
            net->layers[l].input_size * net->layers[l].output_size;
        thread_gradients[l] = calloc(weights_size, sizeof(float));
        thread_bias_gradients[l] =
            calloc(net->layers[l].output_size, sizeof(float));
      }

// Parallel minibatch processing
#pragma omp for schedule(dynamic)
      for (int b = 0; b < train_size; b += batch_size) {
        float batch_loss = 0;

        // Process minibatch
        for (int i = b; i < b + batch_size && i < train_size; i++) {
          float img[INPUT_SIZE];
          float target[OUTPUT_SIZE];

          normalize_images(&data.images[i * INPUT_SIZE], img, 1);
          memset(target, 0, sizeof(target));
          target[data.labels[i]] = 1.0f;

          compute_gradients(net, img, target, thread_gradients,
                            thread_bias_gradients);

          float *output = net->layers[net->num_layers - 1].output;
          batch_loss += -logf(output[data.labels[i]] + 1e-10f);
        }

// Synchronized weight update, TODO this is the largest bottleneck
#pragma omp critical
        {
          apply_updates(net, thread_gradients, thread_bias_gradients,
                        batch_size);

          // Reset gradients
          for (int l = 0; l < net->num_layers; l++) {
            memset(thread_gradients[l], 0,
                   net->layers[l].input_size * net->layers[l].output_size *
                       sizeof(float));
            memset(thread_bias_gradients[l], 0,
                   net->layers[l].output_size * sizeof(float));
          }
        }

        total_loss += batch_loss;
      }

      // Free thread-local memory
      for (int l = 0; l < net->num_layers; l++) {
        free(thread_gradients[l]);
        free(thread_bias_gradients[l]);
      }
      free(thread_gradients);
      free(thread_bias_gradients);
    }

    // Validation (serial)
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

  // Convert and predict
  for (int k = 0; k < INPUT_SIZE; k++)
    img[k] = sample_img[k] / 255.0f;

  int predicted = predict(net, img);
  printf("Predicted Label: %d\n", predicted);

  save_Network(net, "parallel-test.nn");

  free_network(net);
  free(data.images);
  free(data.labels);

  return 0;
}
