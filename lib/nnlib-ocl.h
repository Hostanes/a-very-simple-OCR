
#include <CL/cl.h>

typedef struct {
  int input_size;
  int output_size;

  cl_mem weights;
  cl_mem biases;
  cl_mem weight_momentum;
  cl_mem bias_momentum;

  cl_mem input;      // input to this layer (from previous layer)
  cl_mem activation; // output after activation
  cl_mem gradient;   // gradient for this layer
} Layer_t;

typedef struct {
  int num_layers;
  Layer_t *layers;

  cl_context context;
  cl_command_queue queue;
  cl_program program;

  // Kernels from kernels.cl
  cl_kernel forward_relu_kernel;
  cl_kernel forward_softmax_kernel;
  cl_kernel backward_relu_kernel;
  cl_kernel backward_softmax_kernel;

  // Host-side buffer to read final output
  float *host_output;
} NeuralNetwork_t;

Layer_t create_Layer(cl_context context, int input_Size, int output_Size);

void free_Layer(Layer_t *layer);

NeuralNetwork_t create_NeuralNetwork(cl_context context, int *layer_sizes,
                                     int num_layers);

void free_NeuralNetwork(NeuralNetwork_t *net);

void forward(NeuralNetwork_t *net, cl_command_queue queue, const float *input,
             int input_size, float *output, int output_size);

void backward(NeuralNetwork_t *net, cl_command_queue queue, const float *target,
              int target_size, float learning_rate, float momentum);

void train(NeuralNetwork_t *net, cl_command_queue queue, const float *input,
           int input_size, const float *target, int target_size,
           float learning_rate, float momentum);
