#ifndef NNLIB_H
#define NNLIB_H

#include <CL/cl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC_NUMBER 143
#define FILE_VERSION 1
#define MAX_ACTIVATION_NAME_LEN 32

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
  float *weights;
  float *biases;
  float *weight_momentum;
  float *bias_momentum;
  float *output;
  float *input;

  // Device buffers
  cl_mem weights_buf;
  cl_mem biases_buf;
  cl_mem weight_momentum_buf;
  cl_mem bias_momentum_buf;
  cl_mem output_buf;
  cl_mem input_buf;

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
  int version;
  cl_context context; // Store OpenCL context
} NeuralNetwork_t;

// Initialization
NeuralNetwork_t *create_network(int *layer_sizes, int num_layers,
                                ActivationFunc *activations,
                                ActivationDerivative *derivatives,
                                float learning_rate, float momentum,
                                cl_context context);
void free_network(NeuralNetwork_t *net);

// Device memory management
void upload_network_to_device(NeuralNetwork_t *net, cl_command_queue queue);
void download_network_from_device(NeuralNetwork_t *net, cl_command_queue queue);

// Forward/backward passes
float *forward_pass(NeuralNetwork_t *net, float *input, cl_command_queue queue,
                    cl_program program, int read_output);
void backward_pass(NeuralNetwork_t *net, float *target, cl_command_queue queue,
                   cl_program program);

// Training interface
void train(NeuralNetwork_t *net, float *input, float *target,
           cl_command_queue queue, cl_program program);
int predict(NeuralNetwork_t *net, float *input, cl_command_queue queue,
            cl_program program);

// Utility functions
float calculate_loss(NeuralNetwork_t *net, float *output, float *target);
void compute_gradients(NeuralNetwork_t *net, float *input, float *target,
                       float **gradients, float **bias_gradients);
void apply_updates(NeuralNetwork_t *net, float **gradients,
                   float **bias_gradients, int batch_size);

// Activation functions
float relu(float x);
float relu_derivative(float x);
float linear(float x);
float linear_derivative(float x);
void softmax(float *array, int size);
void softmax_derivative(float *output, float *gradient, int size);
float softmax_placeholder(float x);

// Model persistence
int save_Network(NeuralNetwork_t *network, const char *filename);
NeuralNetwork_t *load_Network(const char *filename, cl_context context);

#endif // NNLIB_H
