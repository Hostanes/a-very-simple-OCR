#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "lib/matrix-math.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef void (*ActivationFunc)(Matrix *);

typedef struct {
  struct Matrix *weights;
  struct Matrix *biases;

  /*
    function pointers for forward act and backward funcs
 */
  ActivationFunc activation;
  ActivationFunc activation_derivative;

  int input_size;
  int output_size;
} Layer_t;

typedef struct {
  Layer_t **layers;
  int num_layers;
  double learning_rate;
} Model_t;

Model_t *create_model(int num_layers, const int *layer_sizes,
                      double learning_rate);
void free_model(Model_t *model);
void forward_pass(Model_t *model, struct Matrix *input,
                  struct Matrix **activations);
void backward_pass(Model_t *model, struct Matrix **activations,
                   struct Matrix *target);

void train_model_batch(Model_t *model, Matrix **inputs, Matrix **targets,
                       int batch_size, int epochs, double learning_rate);

void relu(struct Matrix *mat);
void relu_derivative(struct Matrix *mat);
void softmax(struct Matrix *mat);

double xavier_scale(int input_size);
void xavier_init(struct Matrix *weights);
void random_bias_init(struct Matrix *biases);

#endif // NEURAL_NETWORK_H
