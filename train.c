
#define USE_MNIST_LOADER
#ifndef MNIST_DOUBLE
#define MNIST_DOUBLE

#include "cnn.h"
#include "mnist-dataloader.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_EPOCHS 10
#define LEARNING_RATE 0.001f
#define BATCH_SIZE 32
#define TRAIN_SAMPLES 60000
#define TEST_SAMPLES 10000

// Helper function to find index of maximum value
int argmax(float *array, int size) {
  int max_idx = 0;
  for (int i = 1; i < size; i++) {
    if (array[i] > array[max_idx]) {
      max_idx = i;
    }
  }
  return max_idx;
}

// Convert MNIST data to one-hot encoded labels
void to_one_hot(unsigned int label, float *one_hot) {
  for (int i = 0; i < 10; i++) {
    one_hot[i] = (i == label) ? 1.0f : 0.0f;
  }
}

// Create and initialize the CNN model
CNNModel_t *create_mnist_model() {
  CNNModel_t *model = create_Model(LEARNING_RATE);

  // Architecture similar to LeNet-5
  LayerConfig_t conv1 = {.type = CONV_LAYER,
                         .activation = RELU,
                         .kernel_size = 5,
                         .filters = 6,
                         .padding = 0};
  add_Layer(model, conv1);

  LayerConfig_t pool1 = {.type = MAXPOOL_LAYER, .kernel_size = 2};
  add_Layer(model, pool1);

  LayerConfig_t conv2 = {.type = CONV_LAYER,
                         .activation = RELU,
                         .kernel_size = 5,
                         .filters = 16,
                         .padding = 0};
  add_Layer(model, conv2);

  LayerConfig_t pool2 = {.type = MAXPOOL_LAYER, .kernel_size = 2};
  add_Layer(model, pool2);

  LayerConfig_t dense1 = {
      .type = DENSE_LAYER, .activation = RELU, .neurons = 120};
  add_Layer(model, dense1);

  LayerConfig_t dense2 = {
      .type = DENSE_LAYER, .activation = RELU, .neurons = 84};
  add_Layer(model, dense2);

  LayerConfig_t output = {
      .type = DENSE_LAYER, .activation = SOFTMAX, .neurons = 10};
  add_Layer(model, output);

  return model;
}

void train(CNNModel_t *model, mnist_data *train_data, int num_samples) {
  float total_loss = 0;
  int batches = num_samples / BATCH_SIZE;

  for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    total_loss = 0;

    // Shuffle training data
    for (int i = 0; i < num_samples; i++) {
      int j = i + rand() % (num_samples - i);
      mnist_data temp = train_data[i];
      train_data[i] = train_data[j];
      train_data[j] = temp;
    }

    for (int batch = 0; batch < batches; batch++) {
      // Process batch
      for (int i = 0; i < BATCH_SIZE; i++) {
        int idx = batch * BATCH_SIZE + i;

        // Create input matrix (normalize to 0-1)
        Matrix_t input = {.data = (float *)malloc(28 * 28 * sizeof(float)),
                          .rows = 28,
                          .columns = 28,
                          .channels = 1};

        for (int j = 0; j < 28 * 28; j++) {
          input.data[j] = train_data[idx].data[j] / 255.0f;
        }

        // Create one-hot target
        float target[10];
        to_one_hot(train_data[idx].label, target);

        Matrix_t target_mat = {
            .data = target, .rows = 1, .columns = 10, .channels = 1};

        // Forward and backward pass
        model->forward(model, &input);
        float loss;
        model->backward(model, &target_mat, &loss);
        total_loss += loss;

        free(input.data);
      }

      // Update weights after each batch
      model->update(model);
    }

    printf("Epoch %d, Loss: %.4f\n", epoch + 1, total_loss / batches);
  }
}

float evaluate(CNNModel_t *model, mnist_data *test_data, int num_test) {
  int correct = 0;

  for (int i = 0; i < num_test; i++) {
    // Create input matrix
    Matrix_t input = {.data = (float *)malloc(28 * 28 * sizeof(float)),
                      .rows = 28,
                      .columns = 28,
                      .channels = 1};

    for (int j = 0; j < 28 * 28; j++) {
      input.data[j] = test_data[i].data[j] / 255.0f;
    }

    // Forward pass
    model->forward(model, &input);

    // Get prediction
    CNNLayer_t *output_layer = model->layers;
    while (output_layer->next != NULL) {
      output_layer = output_layer->next;
    }

    int pred = argmax(output_layer->cache.output->data, 10);
    if (pred == test_data[i].label) {
      correct++;
    }

    free(input.data);
  }

  return (float)correct / num_test;
}

int main() {
  srand(time(NULL));

  // Load MNIST data
  mnist_data *train_data = NULL;
  mnist_data *test_data = NULL;
  unsigned int train_count, test_count;

  if (mnist_load("dataset/train-images.idx3-ubyte",
                 "dataset/train-labels.idx1-ubyte", &train_data,
                 &train_count)) {
    printf("Error loading training data\n");
    return 1;
  }

  if (mnist_load("dataset/t10k-images.idx3-ubyte",
                 "dataset/t10k-labels.idx1-ubyte", &test_data, &test_count)) {
    printf("Error loading test data\n");
    return 1;
  }

  printf("Loaded %d training and %d test samples\n", train_count, test_count);

  // Create and initialize model
  CNNModel_t *model = create_mnist_model();

  // Initialize weights (need to do one forward pass first to set input
  // dimensions)
  Matrix_t dummy_input = {.data = (float *)calloc(28 * 28, sizeof(float)),
                          .rows = 28,
                          .columns = 28,
                          .channels = 1};
  model->forward(model, &dummy_input);
  free(dummy_input.data);

  init_Weights(model);

  // Train the model
  printf("Starting training...\n");
  train(model, train_data, train_count);

  // Evaluate
  printf("Evaluating...\n");
  float accuracy = evaluate(model, test_data, test_count);
  printf("Test Accuracy: %.2f%%\n", accuracy * 100);

  // Cleanup
  free_Model(model);
  free(train_data);
  free(test_data);

  return 0;
}

#endif
