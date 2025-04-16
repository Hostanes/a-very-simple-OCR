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

typedef float (*ActivationFunc)(float);
typedef float (*ActivationDerivative)(float);

typedef struct {
  float *weights;
  float *biases;
  float *weight_momentum;
  float *bias_momentum;
  float *output; // after activation
  float *input;  // before activation
  int input_size;
  int output_size;
  ActivationFunc activation;
  ActivationDerivative activation_derivative;
} Layer_t;

typedef struct {
  Layer_t *layers;
  int num_layers;
  float learning_rate;
  float momentum;
  int version; // differentiate between different model files
} NeuralNetwork_t;

// NN INIT, FREE FUNCS
NeuralNetwork_t *create_network(int *layer_sizes, int num_layers,
                                ActivationFunc *activations,
                                ActivationDerivative *derivatives,
                                float learning_rate, float momentum);
void free_network(NeuralNetwork_t *net);

// PROP FUNCS
float *forward_pass(NeuralNetwork_t *net, float *input);
void backward_pass(NeuralNetwork_t *net, float *input, float *target);

// TRAINING FUNCS
void train(NeuralNetwork_t *net, float *input, float *target);
int predict(NeuralNetwork_t *net, float *input);
float calculate_loss(NeuralNetwork_t *net, float *output, float *target);

// alternative function to train, used for parallelized gradients, each thread
// has private gradients
void compute_gradients(NeuralNetwork_t *net, float *input, float *target,
                       float **gradients, float **bias_gradients);
void apply_updates(NeuralNetwork_t *net, float **gradients,
                   float **bias_gradients, int batch_size);

// UTIL FUNCS
void initialize_layer(Layer_t *layer, int input_size, int output_size,
                      ActivationFunc activation,
                      ActivationDerivative derivative);
void randomize_weights(float *weights, int size, float scale);

// ACTIVATION FUNCS
float sigmoid(float x);
float sigmoid_derivative(float x);
float relu(float x);
float relu_derivative(float x);
float linear(float x);
float linear_derivative(float x);
void softmax(float *array, int size);
void softmax_derivative(float *output, float *gradient, int size);
float softmax_placeholder(float x);

// writing mode to file

int save_Network(NeuralNetwork_t *network, char *filename);
NeuralNetwork_t *load_Network(const char *filename);

#endif // NNLIB_H
