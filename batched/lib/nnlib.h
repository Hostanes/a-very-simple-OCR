
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
#define DEFAULT_BATCH_SIZE 32

// Add activation function type enum
typedef enum {
  ACT_RELU,
  ACT_SIGMOID,
  ACT_TANH,
  ACT_LINEAR,
  ACT_SOFTMAX
} ActivationType;

typedef float (*ActivationFunc)(float);
typedef float (*ActivationDerivative)(float);

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

typedef struct {
  Layer_t *layers;
  int num_layers;
  float learning_rate;
  float momentum;
  int version;
  int batch_size; // Current batch size
} NeuralNetwork_t;

// NN INIT, FREE FUNCS
NeuralNetwork_t *create_network(int *layer_sizes, int num_layers,
                                ActivationFunc *activations,
                                ActivationDerivative *derivatives,
                                float learning_rate, float momentum,
                                int batch_size);
void free_network(NeuralNetwork_t *net);
void resize_network_batch(NeuralNetwork_t *net, int new_batch_size);

// PROP FUNCS

float *forward_pass_batch(NeuralNetwork_t *net, float *inputs, int batch_size);

void backward_pass_batch(NeuralNetwork_t *net, float *inputs, float *targets,
                         int batch_size);

// TRAINING FUNCS
void train_batch(NeuralNetwork_t *net, float *inputs, float *targets,
                 int batch_size);

int *predict_batch(NeuralNetwork_t *net, float *inputs, int batch_size);

float calculate_batch_loss(NeuralNetwork_t *net, float *outputs, float *targets,
                           int batch_size);

// UTIL FUNCS
void initialize_layer(Layer_t *layer, int input_size, int output_size,
                      ActivationFunc activation,
                      ActivationDerivative derivative, int batch_size);
void randomize_weights(float *weights, int size, float scale);

// ACTIVATION FUNCS
float sigmoid(float x);
float sigmoid_derivative(float x);
float relu(float x);
float relu_derivative(float x);
float linear(float x);
float linear_derivative(float x);
void softmax_batch(float **arrays, int size, int batch_size);
void softmax_derivative_batch(float **outputs, float **gradients, int size,
                              int batch_size);
float softmax_placeholder(float x);

// FILE IO
int save_Network(NeuralNetwork_t *network, const char *filename);
NeuralNetwork_t *load_Network(const char *filename, int batch_size);

#endif // NNLIB_H
