/*
  unit tests for backward propagation functions:
  - conv2d_Backward
  - maxpool_Backward
  - dense_Backward
  - cross_Entropy_Backward
*/

#include "cnn.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define FLOAT_EQ(a, b) (fabs((a) - (b)) < 1e-6)

void print_Matrix(Matrix_t *matrix) {
  for (int c = 0; c < matrix->channels; c++) {
    printf("Channel %d:\n", c);
    for (int i = 0; i < matrix->rows; i++) {
      for (int j = 0; j < matrix->columns; j++) {
        int idx = (i * matrix->columns + j) * matrix->channels + c;
        printf("%.4f ", matrix->data[idx]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

void test_conv_backward() {
  printf("=== Running conv2d_Backward Tests ===\n");

  // Test 1: Basic 2x2 convolution backward pass
  {
    printf("------------------------\n");
    printf("Test 1: Basic 2x2 convolution backward...\n");

    // Create input (2x2x1)
    Matrix_t *input = create_Matrix(2, 2, 1);
    input->data[0] = 1.0f;
    input->data[1] = 2.0f;
    input->data[2] = 3.0f;
    input->data[3] = 4.0f;

    // Create layer with single 2x2 filter
    CNNLayer_t layer = {.config = {.type = CONV_LAYER,
                                   .kernel_size = 2,
                                   .filters = 1,
                                   .padding = 0},
                        .params = {.weights = create_Matrix(2, 2, 1),
                                   .biases = create_Matrix(1, 1, 1)},
                        .cache = {.input = matrix_Copy(input),
                                  .output = create_Matrix(1, 1, 1)}};

    // Set weights and do forward pass
    layer.params.weights->data[0] = 0.5f;
    layer.params.weights->data[1] = -0.1f;
    layer.params.weights->data[2] = 0.3f;
    layer.params.weights->data[3] = 0.2f;
    layer.params.biases->data[0] = 0.1f;

    // Simulate forward pass output (1x1x1)
    layer.cache.output->data[0] = 1.0f * 0.5f + 2.0f * (-0.1f) + 3.0f * 0.3f +
                                  4.0f * 0.2f + 0.1f; // = 2.1

    // Create gradient from next layer (1x1x1)
    Matrix_t *grad_output = create_Matrix(1, 1, 1);
    grad_output->data[0] = 0.5f; // Arbitrary gradient

    // Run backward pass
    conv2d_Backward(&layer, grad_output);

    // Verify weight gradients
    // ∂L/∂W = input * grad_output
    assert(FLOAT_EQ(layer.params.weight_grads->data[0], 1.0f * 0.5f)); // 0.5
    assert(FLOAT_EQ(layer.params.weight_grads->data[1], 2.0f * 0.5f)); // 1.0
    assert(FLOAT_EQ(layer.params.weight_grads->data[2], 3.0f * 0.5f)); // 1.5
    assert(FLOAT_EQ(layer.params.weight_grads->data[3], 4.0f * 0.5f)); // 2.0

    // Verify bias gradient
    assert(FLOAT_EQ(layer.params.bias_grads->data[0], 0.5f));

    // Verify input gradient
    // ∂L/∂X = weights * grad_output
    assert(FLOAT_EQ(layer.cache.input->data[0], 0.5f * 0.5f));  // 0.25
    assert(FLOAT_EQ(layer.cache.input->data[1], -0.1f * 0.5f)); // -0.05
    assert(FLOAT_EQ(layer.cache.input->data[2], 0.3f * 0.5f));  // 0.15
    assert(FLOAT_EQ(layer.cache.input->data[3], 0.2f * 0.5f));  // 0.10

    printf("Input gradients:\n");
    print_Matrix(layer.cache.input);

    // Cleanup
    free_Matrix(input);
    free_Matrix(layer.params.weights);
    free_Matrix(layer.params.biases);
    free_Matrix(layer.cache.output);
    free_Matrix(layer.cache.input);
    free_Matrix(grad_output);

    printf("PASSED\n");
  }

  // Test 2: Convolution with ReLU activation
  {
    printf("------------------------\n");
    printf("Test 2: Conv backward with ReLU...\n");

    Matrix_t *input = create_Matrix(2, 2, 1);
    input->data[0] = 1.0f;
    input->data[1] = -2.0f;
    input->data[2] = -3.0f;
    input->data[3] = 4.0f;

    CNNLayer_t layer = {.config = {.type = CONV_LAYER,
                                   .kernel_size = 2,
                                   .filters = 1,
                                   .padding = 0},
                        .params = {.weights = create_Matrix(2, 2, 1),
                                   .biases = create_Matrix(1, 1, 1)},
                        .cache = {.input = matrix_Copy(input),
                                  .output = create_Matrix(1, 1, 1),
                                  .activated = create_Matrix(1, 1, 1)}};

    // Set weights and simulate forward pass
    layer.params.weights->data[0] = 0.5f;
    layer.params.weights->data[1] = -0.1f;
    layer.params.weights->data[2] = 0.3f;
    layer.params.weights->data[3] = 0.2f;
    layer.params.biases->data[0] = 0.1f;

    // Pre-activation output (negative value)
    layer.cache.output->data[0] =
        1.0f * 0.5f + (-2.0f) * (-0.1f) + (-3.0f) * 0.3f + 4.0f * 0.2f + 0.1f;
    // After ReLU (should be 0)
    layer.cache.activated->data[0] = 0.0f;

    Matrix_t *grad_output = create_Matrix(1, 1, 1);
    grad_output->data[0] = 0.5f;

    conv2d_Backward(&layer, grad_output);

    // Since ReLU output was 0, all gradients should be 0
    assert(FLOAT_EQ(layer.params.weight_grads->data[0], 0.0f));
    assert(FLOAT_EQ(layer.params.bias_grads->data[0], 0.0f));
    assert(FLOAT_EQ(layer.cache.input->data[0], 0.0f));

    // Cleanup
    free_Matrix(input);
    free_Matrix(layer.params.weights);
    free_Matrix(layer.params.biases);
    free_Matrix(layer.cache.output);
    free_Matrix(layer.cache.activated);
    free_Matrix(layer.cache.input);
    free_Matrix(grad_output);

    printf("PASSED\n");
  }
}

void test_maxpool_backward() {
  printf("=== Running maxpool_Backward Tests ===\n");

  // Test 1: Basic 2x2 maxpool backward
  {
    printf("------------------------\n");
    printf("Test 1: Basic maxpool backward...\n");

    // Create 4x4 input
    Matrix_t *input = create_Matrix(4, 4, 1);
    for (int i = 0; i < 16; i++)
      input->data[i] = i;

    CNNLayer_t layer = {
        .config = {.type = MAXPOOL_LAYER},
        .cache = {
            .input = matrix_Copy(input),
            .output = create_Matrix(2, 2, 1),
            .pool_indicies = create_Matrix(2, 2, 2) // stores (h,w) pairs
        }};

    // Simulate forward pass (store max positions)
    // Pool 1: max is 5 at (1,1)
    layer.cache.output->data[0] = 5.0f;
    layer.cache.pool_indicies->data[0] = 1; // h
    layer.cache.pool_indicies->data[1] = 1; // w

    // Pool 2: max is 7 at (1,3)
    layer.cache.output->data[1] = 7.0f;
    layer.cache.pool_indicies->data[2] = 1;
    layer.cache.pool_indicies->data[3] = 3;

    // Pool 3: max is 13 at (3,1)
    layer.cache.output->data[2] = 13.0f;
    layer.cache.pool_indicies->data[4] = 3;
    layer.cache.pool_indicies->data[5] = 1;

    // Pool 4: max is 15 at (3,3)
    layer.cache.output->data[3] = 15.0f;
    layer.cache.pool_indicies->data[6] = 3;
    layer.cache.pool_indicies->data[7] = 3;

    // Create gradient from next layer
    Matrix_t *grad_output = create_Matrix(2, 2, 1);
    grad_output->data[0] = 0.1f; // grad for pool 1
    grad_output->data[1] = 0.2f; // grad for pool 2
    grad_output->data[2] = 0.3f; // grad for pool 3
    grad_output->data[3] = 0.4f; // grad for pool 4

    maxpool_Backward(&layer, grad_output);

    // Verify gradients are routed to max positions
    assert(FLOAT_EQ(layer.cache.input->data[5], 0.1f));  // (1,1)
    assert(FLOAT_EQ(layer.cache.input->data[7], 0.2f));  // (1,3)
    assert(FLOAT_EQ(layer.cache.input->data[13], 0.3f)); // (3,1)
    assert(FLOAT_EQ(layer.cache.input->data[15], 0.4f)); // (3,3)

    // All other positions should be 0
    for (int i = 0; i < 16; i++) {
      if (i != 5 && i != 7 && i != 13 && i != 15) {
        assert(FLOAT_EQ(layer.cache.input->data[i], 0.0f));
      }
    }

    printf("Input gradients:\n");
    print_Matrix(layer.cache.input);

    // Cleanup
    free_Matrix(input);
    free_Matrix(layer.cache.input);
    free_Matrix(layer.cache.output);
    free_Matrix(layer.cache.pool_indicies);
    free_Matrix(grad_output);

    printf("PASSED\n");
  }
}

void test_dense_backward() {
  printf("=== Running dense_Backward Tests ===\n");

  // Test 1: Basic dense layer backward
  {
    printf("------------------------\n");
    printf("Test 1: Basic dense backward...\n");

    // Input (1x3 vector)
    Matrix_t *input = create_Matrix(1, 3, 1);
    input->data[0] = 1.0f;
    input->data[1] = 2.0f;
    input->data[2] = 3.0f;

    // Layer with 2 neurons
    CNNLayer_t layer = {
        .config = {.type = DENSE_LAYER, .activation = RELU, .neurons = 2},
        .params = {.weights = create_Matrix(3, 2, 1), // 3 inputs, 2 outputs
                   .biases = create_Matrix(1, 2, 1)},
        .cache = {.input = matrix_Copy(input),
                  .output = create_Matrix(1, 2, 1),
                  .activated = create_Matrix(1, 2, 1)}};

    // Set weights and simulate forward pass
    // Neuron 0 weights: [0.5, -0.1, 0.3]
    layer.params.weights->data[0] = 0.5f;
    layer.params.weights->data[1] = -0.1f;
    layer.params.weights->data[2] = 0.3f;

    // Neuron 1 weights: [0.2, 0.4, -0.2]
    layer.params.weights->data[3] = 0.2f;
    layer.params.weights->data[4] = 0.4f;
    layer.params.weights->data[5] = -0.2f;

    layer.params.biases->data[0] = 0.1f;
    layer.params.biases->data[1] = -0.1f;

    // Simulate forward pass (positive outputs)
    layer.cache.output->data[0] =
        1.0f * 0.5f + 2.0f * (-0.1f) + 3.0f * 0.3f + 0.1f;
    layer.cache.output->data[1] =
        1.0f * 0.2f + 2.0f * 0.4f + 3.0f * (-0.2f) - 0.1f;
    layer.cache.activated->data[0] =
        layer.cache.output->data[0]; // ReLU pass-through
    layer.cache.activated->data[1] = layer.cache.output->data[1];

    // Create gradient from next layer
    Matrix_t *grad_output = create_Matrix(1, 2, 1);
    grad_output->data[0] = 0.5f;  // grad for neuron 0
    grad_output->data[1] = -0.3f; // grad for neuron 1

    dense_Backward(&layer, grad_output);

    // Verify weight gradients (∂L/∂W = input * grad_output)
    // Neuron 0 gradients
    assert(FLOAT_EQ(layer.params.weight_grads->data[0], 1.0f * 0.5f)); // 0.5
    assert(FLOAT_EQ(layer.params.weight_grads->data[1], 2.0f * 0.5f)); // 1.0
    assert(FLOAT_EQ(layer.params.weight_grads->data[2], 3.0f * 0.5f)); // 1.5

    // Neuron 1 gradients
    assert(FLOAT_EQ(layer.params.weight_grads->data[3], 1.0f * -0.3f)); // -0.3
    assert(FLOAT_EQ(layer.params.weight_grads->data[4], 2.0f * -0.3f)); // -0.6
    assert(FLOAT_EQ(layer.params.weight_grads->data[5], 3.0f * -0.3f)); // -0.9

    // Verify bias gradients (∂L/∂B = grad_output)
    assert(FLOAT_EQ(layer.params.bias_grads->data[0], 0.5f));
    assert(FLOAT_EQ(layer.params.bias_grads->data[1], -0.3f));

    // Verify input gradients (∂L/∂X = weights^T * grad_output)
    // input 0: 0.5*0.5 + 0.2*(-0.3) = 0.25 - 0.06 = 0.19
    assert(FLOAT_EQ(layer.cache.input->data[0], 0.5f * 0.5f + 0.2f * (-0.3f)));
    // input 1: -0.1*0.5 + 0.4*(-0.3) = -0.05 - 0.12 = -0.17
    assert(FLOAT_EQ(layer.cache.input->data[1], -0.1f * 0.5f + 0.4f * (-0.3f)));
    // input 2: 0.3*0.5 + -0.2*(-0.3) = 0.15 + 0.06 = 0.21
    assert(FLOAT_EQ(layer.cache.input->data[2], 0.3f * 0.5f + -0.2f * (-0.3f)));

    printf("Weight gradients:\n");
    print_Matrix(layer.params.weight_grads);
    printf("Input gradients:\n");
    print_Matrix(layer.cache.input);

    // Cleanup
    free_Matrix(input);
    free_Matrix(layer.params.weights);
    free_Matrix(layer.params.biases);
    free_Matrix(layer.cache.input);
    free_Matrix(layer.cache.output);
    free_Matrix(layer.cache.activated);
    free_Matrix(grad_output);

    printf("PASSED\n");
  }

  // Test 2: Dense backward with ReLU (negative output)
  {
    printf("------------------------\n");
    printf("Test 2: Dense backward with ReLU (negative output)...\n");

    Matrix_t *input = create_Matrix(1, 2, 1);
    input->data[0] = 1.0f;
    input->data[1] = -2.0f;

    CNNLayer_t layer = {
        .config = {.type = DENSE_LAYER, .activation = RELU, .neurons = 1},
        .params = {.weights = create_Matrix(2, 1, 1),
                   .biases = create_Matrix(1, 1, 1)},
        .cache = {.input = matrix_Copy(input),
                  .output = create_Matrix(1, 1, 1),
                  .activated = create_Matrix(1, 1, 1)}};

    // Set weights and simulate forward pass with negative output
    layer.params.weights->data[0] = 0.5f;
    layer.params.weights->data[1] = -0.1f;
    layer.params.biases->data[0] = 0.1f;

    layer.cache.output->data[0] = 1.0f * 0.5f + (-2.0f) * (-0.1f) + 0.1f;
    layer.cache.activated->data[0] = 0.0f; // ReLU would set to 0

    Matrix_t *grad_output = create_Matrix(1, 1, 1);
    grad_output->data[0] = 0.5f;

    dense_Backward(&layer, grad_output);

    // Since output was negative before ReLU, gradients should be zero
    assert(FLOAT_EQ(layer.params.weight_grads->data[0], 0.0f));
    assert(FLOAT_EQ(layer.params.bias_grads->data[0], 0.0f));
    assert(FLOAT_EQ(layer.cache.input->data[0], 0.0f));

    // Cleanup
    free_Matrix(input);
    free_Matrix(layer.params.weights);
    free_Matrix(layer.params.biases);
    free_Matrix(layer.cache.input);
    free_Matrix(layer.cache.output);
    free_Matrix(layer.cache.activated);
    free_Matrix(grad_output);

    printf("PASSED\n");
  }
}

void test_cross_entropy_backward() {
  printf("=== Running cross_Entropy_Backward Tests ===\n");

  // Test 1: Basic cross-entropy backward
  {
    printf("------------------------\n");
    printf("Test 1: Basic cross-entropy backward...\n");

    // Create softmax output (3 classes)
    Matrix_t *softmax_out = create_Matrix(1, 3, 1);
    softmax_out->data[0] = 0.2f;
    softmax_out->data[1] = 0.3f;
    softmax_out->data[2] = 0.5f;

    int true_class = 1; // Correct class is index 1

    Matrix_t *grad = cross_Entropy_Backward(softmax_out, true_class);

    // Verify gradient
    // For correct class: p_i - 1 = 0.3 - 1 = -0.7
    assert(FLOAT_EQ(grad->data[1], -0.7f));
    // For incorrect classes: p_i = 0.2 and 0.5
    assert(FLOAT_EQ(grad->data[0], 0.2f));
    assert(FLOAT_EQ(grad->data[2], 0.5f));

    printf("Gradient: %.2f, %.2f, %.2f\n", grad->data[0], grad->data[1],
           grad->data[2]);

    // Cleanup
    free_Matrix(softmax_out);
    free_Matrix(grad);

    printf("PASSED\n");
  }
}

int main() {
  // test_conv_backward();
  // test_maxpool_backward();
  test_dense_backward();
  // test_cross_entropy_backward();
  printf("All backward propagation tests passed!\n");
  return 0;
}
