
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
