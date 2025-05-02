#ifndef NNLIB_OCL_H
#define NNLIB_OCL_H

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

typedef struct {
  int input_size;
  int output_size;
  cl_mem weights;
  cl_mem biases;
  cl_mem weight_momentum;
  cl_mem bias_momentum;
  cl_mem input;
  cl_mem activation;
  cl_mem gradient;
} Layer_t;

typedef struct {
  int num_layers;
  Layer_t *layers;
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  cl_program program;

  // Kernels for neural network operations
  cl_kernel forward_relu_kernel;
  cl_kernel forward_softmax_kernel;
  cl_kernel backward_relu_kernel;
  cl_kernel backward_softmax_kernel;
  cl_kernel weight_gradient_kernel;
  cl_kernel compute_output_grad_kernel;
} NeuralNetwork_t;

// Function declarations
Layer_t create_Layer(cl_context context, int input_size, int output_size);
void free_Layer(Layer_t *layer);

NeuralNetwork_t create_NeuralNetwork(cl_context context, cl_device_id device,
                                     cl_command_queue queue, int *layer_sizes,
                                     int num_layers);
void free_NeuralNetwork(NeuralNetwork_t *net);

void forward(NeuralNetwork_t *net, cl_command_queue queue, const float *input,
             int input_size, float *output, int output_size);

void backward(NeuralNetwork_t *net, cl_command_queue queue, const float *target,
              int target_size, float learning_rate, float momentum);

void train(NeuralNetwork_t *net, cl_command_queue queue, const float *input,
           int input_size, const float *target, int target_size,
           float learning_rate, float momentum);

#endif // NNLIB_OCL_H
