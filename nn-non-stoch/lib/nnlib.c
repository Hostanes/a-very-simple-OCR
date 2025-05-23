#include "nnlib.h"

#ifndef VERBOSE
#define printf(fmt, ...) (0)
#endif

#define LEAKY_RELU_ALPHA 0.01f

void initialize_network(int *layer_sizes, int num_layers, float **weights,
                        float **biases, float **Z_values, float **A_values,
                        float **dZ_values, float **dW, float **db, float **dA,
                        int **weight_offsets, int **activation_offsets) {

  // Calculate total sizes
  int total_weights = 0, total_biases = 0, total_Z = 0, total_A = 0;
  for (int i = 0; i < num_layers - 1; ++i) {
    total_weights += layer_sizes[i] * layer_sizes[i + 1];
    total_biases += layer_sizes[i + 1];
    total_Z += layer_sizes[i + 1]; // Z values skip input layer
  }
  for (int i = 0; i < num_layers; ++i) {
    total_A += layer_sizes[i]; // A values include input layer
  }

  // Allocate memory
  *weights = (float *)malloc(total_weights * sizeof(float));
  *biases = (float *)malloc(total_biases * sizeof(float));
  *Z_values = (float *)malloc(total_Z * sizeof(float));
  *A_values = (float *)malloc(total_A * sizeof(float));
  *dZ_values = (float *)malloc(total_Z * sizeof(float));
  *dW = (float *)malloc(total_weights * sizeof(float));
  *db = (float *)malloc(total_biases * sizeof(float));
  *dA = (float *)malloc(total_A * sizeof(float));

  // Offset arrays
  *weight_offsets = (int *)malloc((num_layers - 1) * sizeof(int));
  *activation_offsets = (int *)malloc(num_layers * sizeof(int));

  // Fill weight_offsets
  int w_offset = 0;
  for (int i = 0; i < num_layers - 1; ++i) {
    (*weight_offsets)[i] = w_offset;
    w_offset += layer_sizes[i] * layer_sizes[i + 1];
  }

  // Fill activation_offsets (used for A and Z)
  int a_offset = 0;
  for (int i = 0; i < num_layers; ++i) {
    (*activation_offsets)[i] = a_offset;
    a_offset += layer_sizes[i];
  }
}

/*
  forward_Pass
*/
void forward_pass(float *input, float *weights, float *biases, float *Z_values,
                  float *A_values, int *layer_sizes, int num_layers,
                  int *weight_offsets, int *activation_offsets) {

  printf("====================\n");
  printf("Forward pass \n");
  printf("====================\n");

  // Copy input to A[0]
  int input_size = layer_sizes[0];
  memcpy(&A_values[activation_offsets[0]], input, input_size * sizeof(float));
  printf("Input:\n");
  for (int i = 0; i < input_size; ++i) {
    printf("  A[0][%d] = %.4f\n", i, input[i]);
  }

  for (int l = 1; l < num_layers; ++l) {
    int prev_size = layer_sizes[l - 1];
    int curr_size = layer_sizes[l];

    float *A_prev = &A_values[activation_offsets[l - 1]];
    float *Z_curr = &Z_values[activation_offsets[l - 1]]; // reuse offset
    float *A_curr = &A_values[activation_offsets[l]];
    float *b_curr = &biases[activation_offsets[l - 1]];
    float *W_curr = &weights[weight_offsets[l - 1]];

    printf("\nLayer %d: (%d -> %d)\n", l, prev_size, curr_size);

    for (int i = 0; i < curr_size; ++i) {

      printf("  Neuron: %d\n", i);
      float z = b_curr[i];
      printf("  Z[%d][%d] starts with bias %.4f\n", l, i, b_curr[i]);

      for (int j = 0; j < prev_size; ++j) {
        float w = W_curr[i * prev_size + j];
        float a = A_prev[j];
        z += w * a;
        printf("    + W[%d][%d,%d] * A[%d][%d] = %.4f * %.4f\n", l, i, j, l - 1,
               j, w, a);
      }

      Z_curr[i] = z;
      if (l == num_layers - 1) {
        A_curr[i] = z; // will be softmaxed later
        printf("  Z[%d][%d] = %.4f (output pre-softmax)\n", l, i, z);
      } else {
        A_curr[i] = z > 0.0f ? z : LEAKY_RELU_ALPHA * z;
        printf("  Z[%d][%d] = %.4f -> A[%d][%d] (leaky ReLU) = %.4f\n", l, i, z,
               l, i, A_curr[i]);
      }
    }

    // Apply softmax at output layer
    if (l == num_layers - 1) {
      float max_z = A_curr[0];
      for (int i = 1; i < curr_size; ++i)
        if (A_curr[i] > max_z)
          max_z = A_curr[i];

      float sum_exp = 0.0f;
      for (int i = 0; i < curr_size; ++i) {
        A_curr[i] = expf(A_curr[i] - max_z);
        sum_exp += A_curr[i];
      }
      for (int i = 0; i < curr_size; ++i) {
        A_curr[i] /= sum_exp;
        printf("  A[%d][%d] (softmax) = %.4f\n", l, i, A_curr[i]);
      }
    }
  }

  printf("====================\n");
  printf("END Forward pass \n");
  printf("====================\n\n");
}

void backward_pass(float *target, float *weights, float *biases,
                   float *Z_values, float *A_values, float *dZ_values,
                   float *dW, float *db, float *dA, int *layer_sizes,
                   int num_layers, int *weight_offsets,
                   int *activation_offsets) {

  printf("====================\n");
  printf("Backward pass \n");
  printf("====================\n\n");

  int L = num_layers - 1;
  int output_size = layer_sizes[L];
  int A_offset = activation_offsets[L];
  int Z_offset = activation_offsets[L - 1]; // Z for output layer
  int dZ_offset = Z_offset;

  printf("Output layer (Softmax + Cross-Entropy):\n");
  for (int i = 0; i < output_size; ++i) {
    dZ_values[dZ_offset + i] = A_values[A_offset + i] - target[i];
    printf("  dZ[L][%d] = A[%d] - target[%d] = %.4f - %.4f = %.4f\n", i,
           A_offset + i, i, A_values[A_offset + i], target[i],
           dZ_values[dZ_offset + i]);
  }

  printf("\n");

  // Hidden and output layers
  for (int l = num_layers - 1; l > 0; --l) {
    int curr_size = layer_sizes[l];
    int prev_size = layer_sizes[l - 1];

    int A_prev_offset = activation_offsets[l - 1];
    int A_curr_offset = activation_offsets[l];
    int Z_curr_offset = activation_offsets[l - 1]; // for dZ
    int dZ_curr_offset = Z_curr_offset;
    int dA_prev_offset = A_prev_offset;
    int weight_offset = weight_offsets[l - 1];

    float *dZ = &dZ_values[dZ_curr_offset];
    float *A_prev = &A_values[A_prev_offset];
    float *Z_curr = &Z_values[Z_curr_offset];

    float *W = &weights[weight_offset];
    float *dW_curr = &dW[weight_offset];
    float *db_curr = &db[Z_curr_offset];
    float *dA_prev = &dA[dA_prev_offset];

    printf("Layer %d (curr_size=%d, prev_size=%d)\n", l, curr_size, prev_size);

    // dW = dZ * A_prev^T
    printf("  dW = dZ * A_prev^T:\n");
    for (int i = 0; i < curr_size; ++i) {
      for (int j = 0; j < prev_size; ++j) {
        dW_curr[i * prev_size + j] = dZ[i] * A_prev[j];
        printf("    dW[%d][%d] = dZ[%d] * A_prev[%d] = %.4f * %.4f = %.4f\n", i,
               j, i, j, dZ[i], A_prev[j], dW_curr[i * prev_size + j]);
      }
    }

    // db = dZ
    printf("  db = dZ:\n");
    for (int i = 0; i < curr_size; ++i) {
      db_curr[i] = dZ[i];
      printf("    db[%d] = dZ[%d] = %.4f\n", i, i, db_curr[i]);
    }

    // dA_prev = W^T * dZ
    printf("  dA_prev = W^T * dZ:\n");
    for (int j = 0; j < prev_size; ++j) {
      float grad = 0.0f;
      for (int i = 0; i < curr_size; ++i) {
        grad += W[i * prev_size + j] * dZ[i];
        printf("    += W[%d][%d] * dZ[%d] = %.4f * %.4f\n", i, j, i,
               W[i * prev_size + j], dZ[i]);
      }
      dA_prev[j] = grad;
      printf("    dA_prev[%d] = %.4f\n", j, grad);
    }

    // dZ[l-1] = dA[l-1] * ReLU'(Z[l-1])
    if (l > 1) {
      int prev_Z_offset = activation_offsets[l - 2];
      int prev_dZ_offset = prev_Z_offset;
      int prev_dA_offset = activation_offsets[l - 1];

      float *dZ_prev = &dZ_values[prev_dZ_offset];
      float *Z_prev = &Z_values[prev_Z_offset];
      float *dA_curr = &dA[prev_dA_offset];

      printf("  dZ[l-1] = dA[l-1] * ReLU'(Z[l-1]):\n");
      for (int i = 0; i < prev_size; ++i) {
        float z = Z_prev[i];
        float relu_grad = (z > 0.0f) ? 1.0f : LEAKY_RELU_ALPHA;
        dZ_prev[i] = dA_curr[i] * relu_grad;
        printf("    dZ_prev[%d] = dA[%d] * ReLU'(%.4f) = %.4f * %.4f = %.4f\n",
               i, i, z, dA_curr[i], relu_grad, dZ_prev[i]);
      }
    }

    printf("\n");
  }

  printf("====================\n");
  printf("END Backward pass \n");
  printf("====================\n\n");
}

void update_weights(float *weights, float *biases, float *dZ_values,
                    float *A_values, int *layer_sizes, int num_layers,
                    int *weight_offsets, int *activation_offsets,
                    float learning_rate) {

  printf("====================\n");
  printf("Update weights\n");
  printf("====================\n\n");

  for (int l = 1; l < num_layers; ++l) {
    int input_size = layer_sizes[l - 1];
    int output_size = layer_sizes[l];
    int weight_offset = weight_offsets[l - 1];
    int bias_offset = activation_offsets[l - 1]; // biases share Z offset
    int A_prev_offset = activation_offsets[l - 1];
    int dZ_offset = activation_offsets[l - 1];

    float *W = &weights[weight_offset];
    float *b = &biases[bias_offset];
    float *A_prev = &A_values[A_prev_offset];
    float *dZ = &dZ_values[dZ_offset];

    // Update weights: W -= learning_rate * dZ * A_prev^T
    for (int i = 0; i < output_size; ++i) {
      for (int j = 0; j < input_size; ++j) {
        W[i * input_size + j] -= learning_rate * dZ[i] * A_prev[j];

        printf("W[%d,%d] -= %f * %f * %f\n", i, j, learning_rate, dZ[i],
               A_prev[j]);
      }
    }

    // Update biases: b -= learning_rate * dZ
    for (int i = 0; i < output_size; ++i) {
      b[i] -= learning_rate * dZ[i];
    }
  }

  printf("====================\n");
  printf("END Update weights\n");
  printf("====================\n\n");
}
