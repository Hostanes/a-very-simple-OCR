
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for memcpy

void initialize_network(int *layer_sizes, int num_layers, float **weights,
                        float **biases, float **Z_values, float **A_values,
                        float **dZ_values, float **dW, float **db, float **dA,
                        int **weight_offsets, int **activation_offsets);

void forward_pass(float *input, float *weights, float *biases, float *Z_values,
                  float *A_values, int *layer_sizes, int num_layers,
                  int *weight_offsets, int *activation_offsets);

void backward_pass(float *target, float *weights, float *biases,
                   float *Z_values, float *A_values, float *dZ_values,
                   float *dW, float *db, float *dA, int *layer_sizes,
                   int num_layers, int *weight_offsets,
                   int *activation_offsets);

void update_weights(float *weights, float *biases, float *dZ_values,
                    float *A_values, int *layer_sizes, int num_layers,
                    int *weight_offsets, int *activation_offsets,
                    float learning_rate);
