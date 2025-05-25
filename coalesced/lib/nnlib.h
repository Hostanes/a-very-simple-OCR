

#ifndef NNLIB_H
#define NNLIB_H

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// neural network magic number, used at the top of the model file
#define MAGIC_NUMBER 143 // 0x8f
#define FILE_VERSION 1
#define MAX_ACTIVATION_NAME_LEN 32

typedef enum {
  ACT_RELU,
  ACT_SOFTMAX,
  ACT_TANH,
  ACT_LINEAR,
} ActivationType;

typedef float (*ActivationFunc)(float);
typedef float (*ActivationDerivative)(float);

/*
typedef struct {
  float *weights;         // Weight matrix (input_size x output_size)
  float *biases;          // Bias vector (output_size)
  float *weight_momentum; // Momentum for weights
  float *bias_momentum;   // Momentum for biases
  float *outputs;         // Outputs for each sample in batch (after activation)
  float *inputs;          // Inputs for each sample in batch (before activation)
  int input_size;
  int output_size;
  int batch_size;
  ActivationFunc activation;
  ActivationDerivative activation_derivative;
} Layer_t;
*/

typedef struct {

  float learning_Rate;
  float momentum;

  int input_Size; // size of one input

  int *layer_Sizes;
  int num_layers;

  // Activation functions
  ActivationFunc *activations; // Array of activation functions per layer
  ActivationDerivative *activation_derivs; // Corresponding derivatives

  /*
    Weights:
    stored in row major order
  */
  float *weights;
  int num_Of_Weights;  // sum i = 1 -> num layers
                       // += [layer_Sizes[i] * layer_Sizes[i-1]
  int *weight_Offsets; // starts at 0

  /*
    Biases:
  */
  float *biases;
  int num_Of_Biases;
  int *bias_Offsets;

  /*
    Z values:
    preactivation values Z = W.X + b
  */
  float *Z_values;
  int num_Of_Z;
  int *Z_Offsets;

  /*
    A values:
    post activation values, also used for
    the inputs into the first hidden layer
  */
  float *A_values;
  int num_Of_A;
  int *A_Offsets;

  float *weight_gradients; // Same structure as weights
  float *bias_gradients;   // Same structure as biases

  float *weight_Momentum;
  // same size as weights
  // same offsets as weights

  float *bias_Momentum;
  // same size as biases
  // same offsets as biases

  int version;
} NeuralNetwork_t;

// NETWORK CREATE AND FREE
NeuralNetwork_t *create_network(int input_size, int *layer_sizes,
                                int num_layers, ActivationFunc *activations,
                                ActivationDerivative *derivs,
                                float learning_rate, float momentum);
void free_network(NeuralNetwork_t *net);

// UTIL

// PROP FUNCS
void forward_Pass(NeuralNetwork_t *net, float *input);
void backward_Pass(NeuralNetwork_t *net, float *input, float *target);

// ACTIVATION FUNCS
float relu(float x);
float relu_derivative(float x);
void softmax(float *array, int size);
void softmax_derivative(float *output, float *gradient, int size);
float softmax_placeholder(float x);

#endif // NNLIB_H
