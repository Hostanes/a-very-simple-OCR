
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void forward_Pass(float *batch, float *weights, float *biases,
                  float *neuron_Values, int batch_Size, int *layer_Sizes,
                  int num_Layers);

void backward_Pass(float *batch, float *weights, float *neuron_Values,
                   float *gradient_weights, float *gradient_biases,
                   float *output_errors, int batch_Size, int *layer_Sizes,
                   int num_Layers);

void update_Weights(float *weights, float *biases,
                    const float *gradient_weights, const float *gradient_biases,
                    float learning_rate, const int *layer_Sizes,
                    int num_Layers);
