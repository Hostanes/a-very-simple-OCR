
#include "lib/nnlib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IMG_SIZE 16 // Using smaller images for this simple test
#define NUM_CLASSES 2
#define TRAIN_SAMPLES 1000
#define TEST_SAMPLES 200
#define LEARNING_RATE 0.01f
#define EPOCHS 10
#define BATCH_SIZE 32

int layer_Sizes[] = {IMG_SIZE * IMG_SIZE, 32,
                     NUM_CLASSES}; // Simple architecture

void generate_synthetic_data(float *data, float *targets, int num_samples) {
  for (int i = 0; i < num_samples; i++) {
    // Randomly decide if this sample will be left or right
    int is_left = rand() % 2;
    targets[i * NUM_CLASSES + 0] = is_left ? 1.0f : 0.0f;
    targets[i * NUM_CLASSES + 1] = is_left ? 0.0f : 1.0f;

    // Generate image with random noise on the chosen side
    for (int y = 0; y < IMG_SIZE; y++) {
      for (int x = 0; x < IMG_SIZE; x++) {
        int pos = i * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x;

        if ((is_left && x < IMG_SIZE / 2) || (!is_left && x >= IMG_SIZE / 2)) {
          // Add content to the chosen side
          data[pos] = (rand() % 100) / 100.0f;
        } else {
          // Empty side
          data[pos] = 0.0f;
        }
      }
    }
  }
}

void print_simple_image(float *image) {
  for (int y = 0; y < IMG_SIZE; y++) {
    for (int x = 0; x < IMG_SIZE; x++) {
      printf(image[y * IMG_SIZE + x] > 0.5f ? "#" : ".");
    }
    printf("\n");
  }
}

float compute_accuracy(float *data, float *targets, float *weights,
                       float *biases, float *neuron_Values, int num_samples,
                       int batch_size, int *layer_Sizes, int num_Layers) {
  int correct = 0;
  int output_size = layer_Sizes[num_Layers - 1];

  for (int i = 0; i < num_samples; i += batch_size) {
    int current_batch =
        (i + batch_size > num_samples) ? num_samples - i : batch_size;

    forward_Pass(data + i * IMG_SIZE * IMG_SIZE, weights, biases, neuron_Values,
                 current_batch, layer_Sizes, num_Layers);

    float *output = neuron_Values + (num_Layers - 1) * batch_size * output_size;

    for (int j = 0; j < current_batch; j++) {
      int pred =
          (output[j * output_size + 0] > output[j * output_size + 1]) ? 0 : 1;
      int true_label = (targets[(i + j) * output_size + 0] >
                        targets[(i + j) * output_size + 1])
                           ? 0
                           : 1;
      if (pred == true_label) {
        correct++;
      }
    }
  }

  return (float)correct / num_samples;
}

int main() {
  srand(time(NULL));

  // Allocate memory
  float *train_data =
      malloc(TRAIN_SAMPLES * IMG_SIZE * IMG_SIZE * sizeof(float));
  float *train_targets = malloc(TRAIN_SAMPLES * NUM_CLASSES * sizeof(float));
  float *test_data = malloc(TEST_SAMPLES * IMG_SIZE * IMG_SIZE * sizeof(float));
  float *test_targets = malloc(TEST_SAMPLES * NUM_CLASSES * sizeof(float));

  // Generate synthetic data
  generate_synthetic_data(train_data, train_targets, TRAIN_SAMPLES);
  generate_synthetic_data(test_data, test_targets, TEST_SAMPLES);

  // Show some examples
  printf("Training examples:\n");
  for (int i = 0; i < 3; i++) {
    printf("\nExample %d (should be %s):\n", i + 1,
           train_targets[i * NUM_CLASSES] > 0.5f ? "LEFT" : "RIGHT");
    print_simple_image(train_data + i * IMG_SIZE * IMG_SIZE);
  }

  // Initialize network
  int num_Layers = sizeof(layer_Sizes) / sizeof(int);
  int num_Weights = 0, num_Biases = 0, values_Size = 0;

  for (int i = 1; i < num_Layers; i++) {
    num_Weights += layer_Sizes[i] * layer_Sizes[i - 1];
    num_Biases += layer_Sizes[i];
    values_Size += layer_Sizes[i] * BATCH_SIZE;
  }

  float *weights = malloc(num_Weights * sizeof(float));
  float *biases = calloc(num_Biases, sizeof(float));
  float *gradients = calloc(num_Weights, sizeof(float));
  float *bias_Gradients = calloc(num_Biases, sizeof(float));
  float *neuron_Values = calloc(values_Size, sizeof(float));
  float *errors = calloc(values_Size, sizeof(float));

  initialize_weights(weights, biases, layer_Sizes, num_Layers);

  // Training loop
  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    float epoch_Loss = 0.0f;

    for (int i = 0; i < TRAIN_SAMPLES; i += BATCH_SIZE) {
      int current_batch =
          (i + BATCH_SIZE > TRAIN_SAMPLES) ? TRAIN_SAMPLES - i : BATCH_SIZE;

      forward_Pass(train_data + i * IMG_SIZE * IMG_SIZE, weights, biases,
                   neuron_Values, current_batch, layer_Sizes, num_Layers);

      float batch_loss =
          compute_loss(neuron_Values, train_targets + i * NUM_CLASSES,
                       current_batch, NUM_CLASSES, num_Layers);
      epoch_Loss += batch_loss * current_batch;

      backward_Pass(train_data + i * IMG_SIZE * IMG_SIZE, weights, biases,
                    neuron_Values, train_targets + i * NUM_CLASSES, gradients,
                    bias_Gradients, errors, current_batch, layer_Sizes,
                    num_Layers, num_Weights, values_Size);

      update_Weights(weights, biases, gradients, LEARNING_RATE, bias_Gradients,
                     num_Weights, num_Biases, BATCH_SIZE);
    }

    epoch_Loss /= TRAIN_SAMPLES;
    float train_acc = compute_accuracy(train_data, train_targets, weights,
                                       biases, neuron_Values, TRAIN_SAMPLES,
                                       BATCH_SIZE, layer_Sizes, num_Layers);
    float test_acc = compute_accuracy(test_data, test_targets, weights, biases,
                                      neuron_Values, TEST_SAMPLES, BATCH_SIZE,
                                      layer_Sizes, num_Layers);

    printf("Epoch %d - Loss: %.4f, Train Acc: %.2f%%, Test Acc: %.2f%%\n",
           epoch + 1, epoch_Loss, train_acc * 100.0f, test_acc * 100.0f);
  }

  // Test some predictions
  printf("\nTesting some predictions:\n");
  for (int i = 0; i < 5; i++) {
    int idx = rand() % TEST_SAMPLES;
    forward_Pass(test_data + idx * IMG_SIZE * IMG_SIZE, weights, biases,
                 neuron_Values, 1, layer_Sizes, num_Layers);

    float *output = neuron_Values + (num_Layers - 1) * NUM_CLASSES;
    int pred = (output[0] > output[1]) ? 0 : 1;
    int true_label =
        (test_targets[idx * NUM_CLASSES] > test_targets[idx * NUM_CLASSES + 1])
            ? 0
            : 1;

    printf("\nTest %d: Predicted %s, Actual %s\n", i + 1,
           pred == 0 ? "LEFT" : "RIGHT", true_label == 0 ? "LEFT" : "RIGHT");
    print_simple_image(test_data + idx * IMG_SIZE * IMG_SIZE);
  }

  // Clean up
  free(train_data);
  free(train_targets);
  free(test_data);
  free(test_targets);
  free(weights);
  free(biases);
  free(gradients);
  free(bias_Gradients);
  free(neuron_Values);
  free(errors);

  return 0;
}
