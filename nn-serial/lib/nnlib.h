
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void forward_Pass(float *batch, float *weights, float *biases,
                  float *neuron_Values, int batch_Size, int *layer_Sizes,
                  int num_Layers);

void backward_Pass(float *batch, float *weights, float *biases,
                   float *neuron_Values, float *targets, float *gradients,
                   float *bias_gradients, float *errors, int batch_Size,
                   int *layer_Sizes, int num_Layers, int num_Of_Weights,
                   int num_Of_Neurons);

void update_Weights(float *weights, float *biases, float *weight_Gradients,
                    float learning_Rate, float *bias_Gradients, int num_Weights,
                    int num_Biases, int batch_Size);

float compute_loss(float *neuron_Values, float *targets, int batch_Size,
                   int output_size, int num_Layers);

void gradient_check(float *batch, float *weights, float *biases,
                    float *neuron_Values, float *targets,
                    float *weight_Gradients, float *bias_Gradients,
                    float *errors, int batch_Size, int *layer_Sizes,
                    int num_Layers, int num_Of_Weights, int num_Of_Neurons);

void initialize_weights(float *weights, float *biases, int *layer_Sizes,
                        int num_Layers);
