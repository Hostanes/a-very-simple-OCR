
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INPUT_SIZE 784 // nb of floats for each input

void forward_Pass(float *batch, float *weights, float *biases,
                  float *neuron_Values, int batch_Size, int *layer_Sizes,
                  int num_Layers);

void backward_Pass(float *batch, float *weights, float *neuron_Values,
                   float *gradient_weights, float *gradient_biases,
                   float *output_errors, int batch_Size, int *layer_Sizes,
                   int num_Layers);
