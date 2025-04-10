/*
  unit tests for the functions:
  - conv2d_Forward
  - maxpool_Forward
  - flatten_Forward
  - dense_Forward with ReLU and Softmax
*/

#include "cnn.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

// TODO
#define FLOAT_EQ(a, b) (fabs((a) - (b)) < 1e-6)

void print_Matrix(Matrix_t *matrix) {
  for (int c = 0; c < matrix->channels; c++) {
    printf("Channel %d:\n", c);
    for (int i = 0; i < matrix->rows; i++) {
      for (int j = 0; j < matrix->columns; j++) {
        int idx = (i * matrix->columns + j) * matrix->channels + c;
        printf("%.2f ", matrix->data[idx]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

void test_forward() {
  printf("=== Running conv2d_Forward Tests ===\n");

  // Test 1: Identity kernel (should preserve input)
  {
    printf("------------------------\n");
    printf("Test 1: Identity kernel... \n");
    Matrix_t *input = create_Matrix(3, 3, 1);
    for (int i = 0; i < 9; i++)
      input->data[i] = i + 1; // 1-9 grid

    CNNLayer_t layer = {.config = {.type = CONV_LAYER,
                                   .kernel_size = 3,
                                   .filters = 1,
                                   .padding = 0},
                        .params = {.weights = create_Matrix(3, 3, 1),
                                   .biases = create_Matrix(1, 1, 1)}};

    // Create identity kernel (center=1, others=0)
    zero_Init(layer.params.weights);
    layer.params.weights->data[4] = 1.0f; // Center weight
    zero_Init(layer.params.biases);

    conv2d_Forward(&layer, input);

    // Output should match input (minus padding)
    assert(layer.cache.output->rows == 1);
    assert(layer.cache.output->columns == 1);
    assert(FLOAT_EQ(layer.cache.output->data[0], 5.0f));

    free_Matrix(input);
    free_Matrix(layer.params.weights);
    free_Matrix(layer.params.biases);
    free_Matrix(layer.cache.output);
    printf("PASSED\n");
  }

  // Test 2: Edge detection with padding
  {
    printf("------------------------\n");
    printf("Test 2: Edge detection with padding...\n");

    // Create vertical line input
    Matrix_t *input = create_Matrix(3, 3, 1);
    for (int i = 0; i < 9; i++) {
      input->data[i] = (i % 3 == 1) ? 1.0f : 0.0f; // Vertical line
    }

    CNNLayer_t layer = {.config = {.type = CONV_LAYER,
                                   .kernel_size = 3,
                                   .filters = 1,
                                   .padding = 1},
                        .params = {.weights = create_Matrix(3, 3, 1),
                                   .biases = create_Matrix(1, 1, 1)}};

    // Sobel vertical kernel
    float sobel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    memcpy(layer.params.weights->data, sobel, sizeof(sobel));
    zero_Init(layer.params.biases);

    conv2d_Forward(&layer, input);

    float expected[9] = {3, 0, -3, 4, 0, -4, 3, 0, -3};
    // Verify
    for (int i = 0; i < 9; i++) {
      if (!FLOAT_EQ(layer.cache.output->data[i], expected[i])) {
        printf("Mismatch at %d: expected %.1f, got %.1f\n", i, expected[i],
               layer.cache.output->data[i]);
      }
      assert(FLOAT_EQ(layer.cache.output->data[i], expected[i]));
    }

    free_Matrix(input);
    free_Matrix(layer.params.weights);
    free_Matrix(layer.params.biases);
    free_Matrix(layer.cache.output);
    printf("PASSED\n");
  }
  // TODO Test 3: Multi-channel input

  // Test 4: Multiple filters
  {
    printf("------------------------\n");
    printf("Test 4: Multiple filters...\n");

    Matrix_t *input = create_Matrix(2, 2, 1);
    input->data[0] = 1.0f;
    input->data[1] = 0.0f;
    input->data[2] = 0.0f;
    input->data[3] = 1.0f;

    CNNLayer_t layer = {.config = {.type = CONV_LAYER,
                                   .kernel_size = 2,
                                   .filters = 2,
                                   .padding = 0},
                        .params = {
                            .weights = create_Matrix(2, 2, 2), // 2 filters
                            .biases = create_Matrix(1, 2, 1)   // 2 biases
                        }};

    // Filter 0: Identity
    layer.params.weights->data[0] = 1.0f;
    layer.params.weights->data[1] = 0.0f;
    layer.params.weights->data[2] = 0.0f;
    layer.params.weights->data[3] = 1.0f;

    // Filter 1: Inverter
    layer.params.weights->data[4] = 0.0f;
    layer.params.weights->data[5] = 1.0f;
    layer.params.weights->data[6] = 1.0f;
    layer.params.weights->data[7] = 0.0f;

    zero_Init(layer.params.biases);

    conv2d_Forward(&layer, input);

    // Assertions
    assert(FLOAT_EQ(layer.cache.output->data[0], 2.0f));
    assert(FLOAT_EQ(layer.cache.output->data[1], 0.0f));

    free_Matrix(input);
    free_Matrix(layer.params.weights);
    free_Matrix(layer.params.biases);
    free_Matrix(layer.cache.output);
    printf("PASSED\n");
  }

  // Test 5: Max Pooling 2x2 on 4x4 input
  {
    printf("------------------------\n");
    printf("Test 5: Max Pooling 2x2...\n");

    // Create 4x4 input with 1 channel
    Matrix_t *input = create_Matrix(4, 4, 1);
    float values[16] = {1, 3, 2, 1, 4, 6, 5, 2, 1, 2, 8, 9, 1, 0, 6, 7};
    for (int i = 0; i < 16; i++) {
      input->data[i] = values[i];
    }

    CNNLayer_t layer = {.config = {.type = MAXPOOL_LAYER}};

    maxpool_Forward(&layer, input);

    // Output should be 2x2x1 (one pooled value per 2x2 block)
    // Expected max values:
    // block (0,0): max(1,3,4,6) = 6
    // block (0,1): max(2,1,5,2) = 5
    // block (1,0): max(1,2,1,0) = 2
    // block (1,1): max(8,9,6,7) = 9

    assert(FLOAT_EQ(layer.cache.output->data[0], 6.0f));
    assert(FLOAT_EQ(layer.cache.output->data[1], 5.0f));
    assert(FLOAT_EQ(layer.cache.output->data[2], 2.0f));
    assert(FLOAT_EQ(layer.cache.output->data[3], 9.0f));

    // Print debug info
    printf("Input:\n");
    print_Matrix(input);

    printf("Max Pool Output:\n");
    print_Matrix(layer.cache.output);

    // Free memory
    free_Matrix(input);
    free_Matrix(layer.cache.output);
    free_Matrix(layer.cache.pool_indicies);

    printf("PASSED\n");
  }

  {
    printf("------------\n");
    printf("Test 6: Flatten forward\n");

    Matrix_t *input = create_Matrix(2, 2, 2);
    input->data[0] = 1.0f; // Channel 0
    input->data[1] = 2.0f;
    input->data[2] = 3.0f;
    input->data[3] = 4.0f;
    input->data[4] = 10.0f; // Channel 1
    input->data[5] = 20.0f;
    input->data[6] = 30.0f;
    input->data[7] = 40.0f;

    CNNLayer_t flatten_layer = {.config = {.type = FLATTER_LAYER}};
    flatten_Forward(&flatten_layer, input);

    print_Matrix(input);
    print_Matrix(flatten_layer.cache.output);

    assert(flatten_layer.cache.output->rows == 1);
    assert(flatten_layer.cache.output->columns == 8);
    assert(flatten_layer.cache.output->channels == 1);

    free_Matrix(input);
    free_Matrix(flatten_layer.cache.output);
    printf("PASSED\n");
  }

  // Test: Dense forward with ReLU and Softmax
  {
    printf("------------------------\n");
    printf("Test: Dense forward with ReLU and Softmax...\n");

    // Input vector: shape (1, 3, 1)
    Matrix_t *input = create_Matrix(1, 3, 1);
    input->data[0] = 1.0f;
    input->data[1] = -2.0f;
    input->data[2] = 3.0f;

    // Weights: shape (2, 3, 1)  -> 2 output neurons, 3 inputs
    Matrix_t *weights = create_Matrix(2, 3, 1);
    weights->data[0] = 1.0f; // neuron 0: [1.0, 0.0, -1.0]
    weights->data[1] = 0.0f;
    weights->data[2] = -1.0f;

    weights->data[3] = -1.0f; // neuron 1: [-1.0, 1.0, 1.0]
    weights->data[4] = 1.0f;
    weights->data[5] = 1.0f;

    // Biases: shape (1, 2, 1)
    Matrix_t *biases = create_Matrix(1, 2, 1);
    biases->data[0] = 0.0f;
    biases->data[1] = 0.0f;

    CNNLayer_t relu_layer = {
        .config = {.type = DENSE_LAYER, .activation = RELU},
        .params = {.weights = weights, .biases = biases}};

    dense_Forward(&relu_layer, input);
    printf("ReLU output: %.2f, %.2f\n", relu_layer.cache.activated->data[0],
           relu_layer.cache.activated->data[1]);

    assert(FLOAT_EQ(relu_layer.cache.activated->data[0],
                    0.0f)); // = -2 → relu = 0
    assert(FLOAT_EQ(relu_layer.cache.activated->data[1],
                    0.0f)); // = 0 → relu = 0

    relu_layer.config.activation = SOFTMAX;
    dense_Forward(&relu_layer, input);

    float a = relu_layer.cache.activated->data[0];
    float b = relu_layer.cache.activated->data[1];
    printf("Softmax output: %.2f, %.2f\n", a, b);
    assert(FLOAT_EQ(a + b, 1.0f));

    // Cleanup
    free_Matrix(input);
    free_Matrix(weights);
    free_Matrix(biases);
    free_Matrix(relu_layer.cache.output);
    free_Matrix(relu_layer.cache.activated);

    printf("PASSED\n");
  }

  printf("All Forward tests passed!\n\n");
}

int main() { test_forward(); }
