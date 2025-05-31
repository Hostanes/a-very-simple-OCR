#include "nnlib.h"

#include "nnlib.h"
#include <stdlib.h>

#ifndef VERBOSE
#define printf(fmt, ...) (0)
#endif

#define ALIGNMENT 32

// =====================================================
//                    HELPER FUNCS
// =====================================================

// Helper function to calculate total size needed for a layer's weights
static int calculate_total_weights(int input_size, int *layer_sizes,
                                   int num_layers) {
  int total = 0;
  for (int i = 0; i < num_layers; i++) {
    int in_dim = (i == 0) ? input_size : layer_sizes[i - 1];
    int out_dim = layer_sizes[i];
    total += in_dim * out_dim;
  }
  return total;
}

// Helper function to calculate total size needed for biases
static int calculate_total_biases(int *layer_sizes, int num_layers) {
  int total = 0;
  for (int i = 1; i <= num_layers; i++) {
    total += layer_sizes[i];
  }
  return total;
}

// Helper function to calculate total Z and A values needed
static int calculate_total_activations(int input_size, int *layer_sizes,
                                       int num_layers) {
  int total = input_size; // Input layer
  for (int i = 0; i < num_layers; i++) {
    total += layer_sizes[i];
  }
  return total;
}

// =====================================================
//                    CREATE FUNCS
// =====================================================

NeuralNetwork_t *create_network(int input_size, int *layer_sizes,
                                int num_layers, ActivationFunc *activations,
                                ActivationDerivative *derivs,
                                float learning_rate, float momentum) {
  // Allocate network structure
  NeuralNetwork_t *net = (NeuralNetwork_t *)malloc(sizeof(NeuralNetwork_t));
  if (!net)
    return NULL;

  // Set basic parameters
  net->learning_Rate = learning_rate;
  net->momentum = momentum;
  net->input_Size = input_size;
  net->num_layers = num_layers;
  net->version = FILE_VERSION;

  // Copy layer sizes
  net->layer_Sizes = (int *)malloc(num_layers * sizeof(int));
  if (!net->layer_Sizes) {
    free(net);
    return NULL;
  }
  memcpy(net->layer_Sizes, layer_sizes, num_layers * sizeof(int));

  // Copy activation functions
  net->activations =
      (ActivationFunc *)malloc(num_layers * sizeof(ActivationFunc));
  net->activation_derivs =
      (ActivationDerivative *)malloc(num_layers * sizeof(ActivationDerivative));
  if (!net->activations || !net->activation_derivs) {
    free(net->layer_Sizes);
    free(net->activations);
    free(net->activation_derivs);
    free(net);
    return NULL;
  }
  memcpy(net->activations, activations, num_layers * sizeof(ActivationFunc));
  memcpy(net->activation_derivs, derivs,
         num_layers * sizeof(ActivationDerivative));

  // Calculate buffer sizes
  net->num_Of_Weights =
      calculate_total_weights(input_size, layer_sizes, num_layers);
  net->num_Of_Biases = calculate_total_biases(layer_sizes, num_layers);
  net->num_Of_Z =
      calculate_total_activations(input_size, layer_sizes, num_layers) -
      input_size;
  net->num_Of_A =
      calculate_total_activations(input_size, layer_sizes, num_layers);

  // Allocate buffers

  // Allocate aligned buffers for all major arrays
  net->weights =
      (float *)aligned_alloc(ALIGNMENT, net->num_Of_Weights * sizeof(float));
  memset(net->weights, 0, net->num_Of_Weights * sizeof(float));

  net->biases =
      (float *)aligned_alloc(ALIGNMENT, net->num_Of_Biases * sizeof(float));
  memset(net->biases, 0, net->num_Of_Biases * sizeof(float));

  net->Z_values =
      (float *)aligned_alloc(ALIGNMENT, net->num_Of_Z * sizeof(float));
  memset(net->Z_values, 0, net->num_Of_Z * sizeof(float));

  net->A_values =
      (float *)aligned_alloc(ALIGNMENT, net->num_Of_A * sizeof(float));
  memset(net->A_values, 0, net->num_Of_A * sizeof(float));

  net->weight_gradients =
      (float *)aligned_alloc(ALIGNMENT, net->num_Of_Weights * sizeof(float));
  memset(net->weight_gradients, 0, net->num_Of_Weights * sizeof(float));

  net->bias_gradients =
      (float *)aligned_alloc(ALIGNMENT, net->num_Of_Biases * sizeof(float));
  memset(net->bias_gradients, 0, net->num_Of_Biases * sizeof(float));

  net->weight_Momentum =
      (float *)aligned_alloc(ALIGNMENT, net->num_Of_Weights * sizeof(float));
  memset(net->weight_Momentum, 0, net->num_Of_Weights * sizeof(float));

  net->bias_Momentum =
      (float *)aligned_alloc(ALIGNMENT, net->num_Of_Biases * sizeof(float));
  memset(net->bias_Momentum, 0, net->num_Of_Biases * sizeof(float));

  // Offset arrays don't need alignment (they're small and accessed
  // sequentially)
  net->weight_Offsets = (int *)calloc(num_layers, sizeof(int));
  net->bias_Offsets = (int *)calloc(num_layers, sizeof(int));
  net->Z_Offsets = (int *)calloc(num_layers, sizeof(int));
  net->A_Offsets = (int *)calloc(num_layers, sizeof(int));

  // Check all allocations succeeded
  if (!net->weights || !net->biases || !net->Z_values || !net->A_values ||
      !net->weight_gradients || !net->bias_gradients || !net->weight_Momentum ||
      !net->bias_Momentum || !net->weight_Offsets || !net->bias_Offsets ||
      !net->Z_Offsets || !net->A_Offsets) {
    free_network(net);
    return NULL;
  }

  // Calculate offsets
  int weight_offset = 0;
  int bias_offset = 0;
  int z_offset = 0;
  int a_offset = input_size; // First 'a_offset' is after input values

  for (int l = 0; l < num_layers; l++) {
    net->weight_Offsets[l] = weight_offset;
    net->bias_Offsets[l] = bias_offset;
    net->Z_Offsets[l] = z_offset;
    net->A_Offsets[l] = a_offset;

    int input_dim = (l == 0) ? input_size : layer_sizes[l - 1];
    int output_dim = layer_sizes[l];

    weight_offset += input_dim * output_dim;
    bias_offset += output_dim;
    z_offset += output_dim;
    a_offset += output_dim;
  }

  // Initialize weights (Xavier initialization)
  for (int l = 0; l < num_layers; l++) {
    int input_dim = (l == 0) ? input_size : layer_sizes[l - 1];
    int output_dim = layer_sizes[l];
    float scale = sqrtf(2.0f / input_dim);

    int offset = net->weight_Offsets[l];
    if (offset < 0 || offset >= net->num_Of_Weights) {
      printf("Invalid weight offset at layer %d: %d\n", l, offset);
      exit(1);
    }

    float *layer_weights = net->weights + offset;
    for (int i = 0; i < input_dim * output_dim; i++) {
      layer_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }
  }

  return net;
}

void free_network(NeuralNetwork_t *net) {
  if (!net)
    return;

  // Free all dynamically allocated arrays
  free(net->layer_Sizes);
  free(net->activations);
  free(net->activation_derivs);
  free(net->weights);
  free(net->biases);
  free(net->Z_values);
  free(net->A_values);
  free(net->weight_gradients);
  free(net->bias_gradients);
  free(net->weight_Momentum);
  free(net->bias_Momentum);
  free(net->weight_Offsets);
  free(net->bias_Offsets);
  free(net->Z_Offsets);
  free(net->A_Offsets);

  // Free the network structure itself
  free(net);
}

// =====================================================
//                    ACTIVATION FUNCS
// =====================================================

// Approximate exp(x) with ~1% relative error using a polynomial and bit hack
inline float fast_exp(float x) {
  union {
    uint32_t i;
    float f;
  } v;

  // Constants based on IEEE float magic (2^23 * log2(e) ≈ 12102203)
  float a = 12102203.0f;
  float b = 1065353216.0f;

  if (x < -100.0f)
    return 0.0f; // prevent underflow

  v.i = (uint32_t)(a * x + b);
  return v.f;
}

float relu(float x) { return x > 0 ? x : 0; }

float relu_derivative(float x) { return x > 0 ? 1 : 0; }

float softmax_placeholder(float x) { return x; }

void softmax(float *array, int size) {

  float max_val = array[0];
#pragma omp parallel for reduction(max : max_val)
  for (int i = 1; i < size; i++) {
    if (array[i] > max_val)
      max_val = array[i];
  }

  float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < size; i++) {
    array[i] = fast_exp(array[i] - max_val);
    sum += array[i];
  }

#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    array[i] /= sum;
  }
}

void softmax_into(const float *input, float *output, int size) {
  // Find the max value for numerical stability
  float max_val = input[0];
#pragma omp parallel for reduction(max : max_val)
  for (int i = 1; i < size; i++) {
    if (input[i] > max_val)
      max_val = input[i];
  }

  // Compute exponentials and their sum
  float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < size; i++) {
    output[i] = fast_exp(input[i] - max_val);
    sum += output[i];
  }

  // Normalize to get probabilities
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    output[i] /= sum;
  }
}

void softmax_derivative(float *output, float *gradient, int size) {
  float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
  for (int j = 0; j < size; j++) {
    sum += output[j] * gradient[j];
  }

#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    gradient[i] = output[i] * (gradient[i] - sum);
  }
}

// =====================================================
//                    PROPAGATION FUNCS
// =====================================================
void forward_Pass(NeuralNetwork_t *net, float *input) {
  printf("\n=== FORWARD PASS ===\n");

  // Copy input to first layer's A values (input layer)
  for (int i = 0; i < net->input_Size; i++) {
    net->A_values[i] = input[i];
  }
  printf("Input values: ");
  for (int i = 0; i < net->input_Size; i++)
    printf("%.4f ", net->A_values[i]);
  printf("\n");

  for (int l = 0; l < net->num_layers; l++) {
    printf("\nLayer %d:\n", l);
    int input_size = (l == 0) ? net->input_Size : net->layer_Sizes[l - 1];
    int output_size = net->layer_Sizes[l];

    float *weights = net->weights + net->weight_Offsets[l];
    float *biases = net->biases + net->bias_Offsets[l];
    float *z_values = net->Z_values + net->Z_Offsets[l];
    float *a_values = net->A_values + net->A_Offsets[l];
    float *prev_a_values =
        (l == 0) ? net->A_values : (net->A_values + net->A_Offsets[l - 1]);

    printf("Weights (first 3): %.4f %.4f %.4f...\n", weights[0], weights[1],
           weights[2]);
    printf("Biases (first 3): %.4f %.4f %.4f...\n", biases[0], biases[1],
           biases[2]);

    // Compute Z = W * A_prev + b
#pragma omp parallel for
    for (int o = 0; o < output_size; o++) {
      float z = biases[o];
      const float *weight_row =
          weights + o; // Point to start of this output's weights

#pragma omp simd reduction(+ : z)
      for (int i = 0; i < input_size; i++) {
        z += weight_row[i * output_size] *
             prev_a_values[i]; // Stride by output_size
      }
      z_values[o] = z;
    }

    printf("Z values (first 3): %.4f %.4f %.4f...\n", z_values[0], z_values[1],
           z_values[2]);

    // Apply activation
    if (l == net->num_layers - 1 &&
        net->activations[l] == softmax_placeholder) {
      softmax_into(z_values, a_values, output_size);
    } else {
#pragma omp parallel for
      for (int o = 0; o < output_size; o++) {
        a_values[o] = net->activations[l](z_values[o]);
      }
    }

    printf("A values (first 3): %.4f %.4f %.4f...\n", a_values[0], a_values[1],
           a_values[2]);
  }
}

void backward_Pass(NeuralNetwork_t *net, float *input, float *target) {
  printf("\n=== BACKWARD PASS ===\n");

  forward_Pass(net, input);

  printf("\n=== BACKWARD PASS ===\n");

  float *errors = (float *)malloc(sizeof(float) * net->num_Of_Z);

  int output_Layer_IDX = net->num_layers - 1;
  float *output_Z = net->Z_values + net->Z_Offsets[output_Layer_IDX];
  float *output_A = net->A_values + net->A_Offsets[output_Layer_IDX];
  float *output_Errors = errors + net->Z_Offsets[output_Layer_IDX];
  int output_Size = net->layer_Sizes[output_Layer_IDX];

  printf("\nOutput layer errors:\n");
  if (net->activations[output_Layer_IDX] == softmax_placeholder) {

#pragma omp parallel for
    for (int i = 0; i < output_Size; i++) {
      output_Errors[i] = (output_A[i] - target[i]);
      printf("Output %d: A=%.4f, Target=%.4f, Error=%.4f\n", i, output_A[i],
             target[i], output_Errors[i]);
    }
  } else {

#pragma omp parallel for
    for (int i = 0; i < output_Size; i++) {
      float deriv = net->activation_derivs[output_Layer_IDX](output_Z[i]);
      output_Errors[i] = (output_A[i] - target[i]) * deriv;
      printf("Output %d: A=%.4f, Target=%.4f, Deriv=%.4f, Error=%.4f\n", i,
             output_A[i], target[i], deriv, output_Errors[i]);
    }
  }

  // Backpropagate errors
  for (int l = net->num_layers - 2; l >= 0; l--) {
    printf("\nBackprop layer %d:\n", l);
    int current_size = net->layer_Sizes[l];
    int next_size = net->layer_Sizes[l + 1];

    float *current_errors = errors + net->Z_Offsets[l];
    float *next_errors = errors + net->Z_Offsets[l + 1];
    float *weights_next = net->weights + net->weight_Offsets[l + 1];
    float *current_Z = net->Z_values + net->Z_Offsets[l];

    printf("Next layer weights (first 3): %.4f %.4f %.4f...\n", weights_next[0],
           weights_next[1], weights_next[2]);

#pragma omp parallel for
    for (int i = 0; i < current_size; i++) {
      float error = 0.0f;
      for (int j = 0; j < next_size; j++) {
        error += weights_next[i * next_size + j] * next_errors[j];
      }
      current_errors[i] = error * net->activation_derivs[l](current_Z[i]);
      printf("Neuron %d: Error=%.4f, Z=%.4f, Deriv=%.4f, FinalError=%.4f\n", i,
             error, current_Z[i], net->activation_derivs[l](current_Z[i]),
             current_errors[i]);
    }
  }

  // Update weights and biases
  for (int l = 0; l < net->num_layers; l++) {
    printf("\nUpdating layer %d:\n", l);
    int input_size = (l == 0) ? net->input_Size : net->layer_Sizes[l - 1];
    int output_size = net->layer_Sizes[l];

    float *layer_weights = net->weights + net->weight_Offsets[l];
    float *layer_biases = net->biases + net->bias_Offsets[l];
    float *layer_Wgrad = net->weight_gradients + net->weight_Offsets[l];
    float *layer_Bgrad = net->bias_gradients + net->bias_Offsets[l];
    float *layer_Wmomentum = net->weight_Momentum + net->weight_Offsets[l];
    float *layer_Bmomentum = net->bias_Momentum + net->bias_Offsets[l];
    float *prev_A =
        (l == 0) ? net->A_values : (net->A_values + net->A_Offsets[l - 1]);
    float *layer_errors = errors + net->Z_Offsets[l];

    printf("Pre-update weights (first 3): %.4f %.4f %.4f...\n",
           layer_weights[0], layer_weights[1], layer_weights[2]);
    printf("Pre-update biases (first 3): %.4f %.4f %.4f...\n", layer_biases[0],
           layer_biases[1], layer_biases[2]);
    printf("Pre-update momentum (first 3): %.4f %.4f %.4f...\n",
           layer_Wmomentum[0], layer_Wmomentum[1], layer_Wmomentum[2]);

    // Calculate gradients

#pragma omp parallel for
    for (int i = 0; i < input_size; i++) {
      for (int j = 0; j < output_size; j++) {
        int idx = i * output_size + j;
        layer_Wgrad[idx] = prev_A[i] * layer_errors[j];
      }
    }
    memcpy(layer_Bgrad, layer_errors, output_size * sizeof(float));

    printf("Weight gradients (first 3): %.4f %.4f %.4f...\n", layer_Wgrad[0],
           layer_Wgrad[1], layer_Wgrad[2]);
    printf("Bias gradients (first 3): %.4f %.4f %.4f...\n", layer_Bgrad[0],
           layer_Bgrad[1], layer_Bgrad[2]);

    // Apply updates

#pragma omp parallel for
    for (int i = 0; i < input_size * output_size; i++) {
      layer_Wmomentum[i] = net->momentum * layer_Wmomentum[i] +
                           net->learning_Rate * layer_Wgrad[i];
      layer_weights[i] -= layer_Wmomentum[i];
      layer_Wgrad[i] = 0.0f;
    }

#pragma omp parallel for
    for (int j = 0; j < output_size; j++) {
      layer_Bmomentum[j] = net->momentum * layer_Bmomentum[j] +
                           net->learning_Rate * layer_Bgrad[j];
      layer_biases[j] -= layer_Bmomentum[j];
      layer_Bgrad[j] = 0.0f;
    }

    printf("Post-update weights (first 3): %.4f %.4f %.4f...\n",
           layer_weights[0], layer_weights[1], layer_weights[2]);
    printf("Post-update biases (first 3): %.4f %.4f %.4f...\n", layer_biases[0],
           layer_biases[1], layer_biases[2]);
    printf("Post-update momentum (first 3): %.4f %.4f %.4f...\n",
           layer_Wmomentum[0], layer_Wmomentum[1], layer_Wmomentum[2]);
  }

  free(errors);
}
