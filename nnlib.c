/*
  nnlib.c

  Implementation file of nnlib.h
*/

#include "nnlib.h"
#include "lib-omp/config.h"
#include "lib/matrix-math.h"

/*
  Creates a dense neural network model,
  uses xavier init for weights
  uses randomized init for biasis

  param num_layers: including input and output layers
  param layer_sizes, an array of ints, each index is the size of that layer
  param learning rate:

  returns *Model_t: pointer to Model_t
*/
Model_t *create_model(int num_layers, const int *layer_sizes,
                      double learning_rate) {
  Model_t *model = (Model_t *)malloc(sizeof(Model_t));
  model->num_layers = num_layers;
  model->learning_rate = learning_rate;
  model->layers = (Layer_t **)malloc(num_layers * sizeof(Layer_t *));

  for (int i = 0; i < num_layers; i++) {
    model->layers[i] = (Layer_t *)malloc(sizeof(Layer_t));
    model->layers[i]->input_size = layer_sizes[i];
    model->layers[i]->output_size = layer_sizes[i + 1];

    // init weights
    model->layers[i]->weights = init_Matrix(layer_sizes[i + 1], layer_sizes[i]);
    xavier_init(model->layers[i]->weights);

    model->layers[i]->biases = init_Matrix(layer_sizes[i + 1], 1);
    random_bias_init(model->layers[i]->biases);

    // set activation functions
    if (i < num_layers - 1) {
      model->layers[i]->activation = relu;
      model->layers[i]->activation_derivative = relu_derivative;
    } else {
      model->layers[i]->activation = softmax;
      model->layers[i]->activation_derivative =
          NULL; // Not used for output layer
    }
  }

  return model;
}

// free model memory
void free_model(Model_t *model) {
  for (int i = 0; i < model->num_layers; i++) {
    matrix_Free(model->layers[i]->weights);
    matrix_Free(model->layers[i]->biases);
    free(model->layers[i]);
  }
  free(model->layers);
  free(model);
}

/* ======================= */
/* Optimized Forward Pass */
/* ======================= */
/*
  Forward pass through the entire network for 1 input

  at each layer:
  output = weights . input + bias
  activation = Act(output), e.g. ReLU or Softmax

*/
void forward_pass(Model_t *model, Matrix *input, Matrix **activations) {
// First activation is the input
#ifdef USE_OMP
#pragma omp parallel for if (input->rows > 1000)
#endif
  for (int i = 0; i < input->rows; i++) {
    activations[0]->data[i][0] = input->data[i][0];
  }

  // Forward through each layer
  for (int l = 0; l < model->num_layers; l++) {
    Layer_t *layer = model->layers[l];

    // Use dot product for matrix multiplication
    Matrix *temp = dot_Mat(layer->weights, activations[l]);

// Add bias (parallelized)
#ifdef USE_OMP
#pragma omp parallel for if (layer->biases->rows > 256)
#endif
    for (int i = 0; i < layer->biases->rows; i++) {
      activations[l + 1]->data[i][0] =
          temp->data[i][0] + layer->biases->data[i][0];
    }

    // Apply activation
    layer->activation(activations[l + 1]);

    matrix_Free(temp); // Free temporary matrix
  }
}
/*
Backward pass (backpropagation) through the entire network for 1 input

At each layer (starting from output → input):
    Compute error derivative: dLoss/dOutput
    Calculate weight gradients: dLoss/dWeights = error . input^T
    Calculate bias gradients: dLoss/dBiases = error
    Propagate error backward: dLoss/dInput = weights^T . error
    Apply activation derivative: error = dLoss/dInput * dActivation/dOutput
    Update weights/biases: weights -= learning_rate * dLoss/dWeights
    biases -= learning_rate * dLoss/dBiases
*/
void backward_pass(Model_t *model, struct Matrix **activations,
                   struct Matrix *target) {
  int output_layer_idx = model->num_layers - 1;
  int output_size = model->layers[output_layer_idx]->output_size;

  Matrix *output_error = init_Matrix(output_size, 1);
  for (int i = 0; i < output_size; i++) {
    output_error->data[i][0] =
        activations[model->num_layers]->data[i][0] - target->data[i][0];
  }

  Matrix *delta = output_error;

  // iterate over each layer backwards, starting from the end
  for (int l = model->num_layers - 1; l >= 0; l--) {
    Matrix *activation_transpose = init_Matrix(1, model->layers[l]->input_size);
    for (int i = 0; i < model->layers[l]->input_size; i++) {
      activation_transpose->data[0][i] = activations[l]->data[i][0];
    }

    Matrix *weight_gradient = dot_Mat(delta, activation_transpose);

    // Clip gradients to [-1.0, 1.0] range
    double max_grad = 1.0;
    for (int i = 0; i < weight_gradient->rows; i++) {
      for (int j = 0; j < weight_gradient->columns; j++) {
        weight_gradient->data[i][j] =
            fmax(fmin(weight_gradient->data[i][j], max_grad), -max_grad);
      }
    }

    // After computing weight_gradient
    double grad_norm = 0.0;
    for (int i = 0; i < weight_gradient->rows; i++) {
      for (int j = 0; j < weight_gradient->columns; j++) {
        grad_norm += fabs(weight_gradient->data[i][j]);
      }
    }

    // printf("Layer %d gradient norm: %.6f\n", l, grad_norm);

    // Update weights
    for (int i = 0; i < model->layers[l]->weights->rows; i++) {
      for (int j = 0; j < model->layers[l]->weights->columns; j++) {
        model->layers[l]->weights->data[i][j] -=
            model->learning_rate * weight_gradient->data[i][j];
      }
    }

    // Update biases
    for (int i = 0; i < model->layers[l]->biases->rows; i++) {
      model->layers[l]->biases->data[i][0] -=
          model->learning_rate * delta->data[i][0];
    }

    if (l > 0) {
      // Backpropagate error to previous layer
      Matrix *weights_transpose = init_Matrix(
          model->layers[l]->weights->columns, model->layers[l]->weights->rows);

      for (int i = 0; i < model->layers[l]->weights->rows; i++) {
        for (int j = 0; j < model->layers[l]->weights->columns; j++) {
          weights_transpose->data[j][i] = model->layers[l]->weights->data[i][j];
        }
      }

      Matrix *prev_error = dot_Mat(weights_transpose, delta);
      Matrix *relu_deriv = init_Matrix(model->layers[l - 1]->output_size, 1);

      for (int i = 0; i < model->layers[l - 1]->output_size; i++) {
        relu_deriv->data[i][0] = activations[l]->data[i][0] > 0 ? 1.0 : 0.0;
      }

      for (int i = 0; i < model->layers[l - 1]->output_size; i++) {
        prev_error->data[i][0] *= relu_deriv->data[i][0];
      }

      matrix_Free(delta);
      delta = prev_error;
      matrix_Free(relu_deriv);
      matrix_Free(weights_transpose);
    }

    matrix_Free(activation_transpose);
    matrix_Free(weight_gradient);
  }

  matrix_Free(delta);
}

/* ======================== */
/* Optimized Training Loop */
/* ======================== */
/*
  Trains model on a full dataset for multiple epochs
  param model Neural network model
  param inputs Array of input matrices (batch)
  param targets Array of target matrices (batch)
  param batch_size Number of samples in batch
  param epochs Number of full passes through the batch
  param learning_rate Learning rate for weight updates
*/
void train_model_batch(Model_t *model, Matrix **inputs, Matrix **targets,
                       int batch_size, int epochs, double learning_rate) {
  double total_start = omp_get_wtime();

  // Initialize thread-private activation buffers
  Matrix *thread_activations[omp_get_max_threads()][model->num_layers + 1];
#ifdef USE_OMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    for (int i = 0; i <= model->num_layers; i++) {
      int size = (i == 0) ? model->layers[0]->input_size
                          : model->layers[i - 1]->output_size;
      thread_activations[tid][i] = init_Matrix(size, 1);
    }
  }

  for (int epoch = 0; epoch < epochs; epoch++) {
    double epoch_start = omp_get_wtime();
    double epoch_loss = 0.0;
    int correct = 0;

    // it actually is possible to parallelize different input samples!!!
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : epoch_loss, correct)                    \
    schedule(dynamic, 100)
#endif

    for (int sample = 0; sample < batch_size; sample++) {
      int tid = omp_get_thread_num();

      // ===============
      // forward with private activations
      copy_Mat(thread_activations[tid][0], inputs[sample]);
      forward_pass(model, thread_activations[tid][0], thread_activations[tid]);

      // ===============
      // accuracy
      int pred = 0;
      double max_val = thread_activations[tid][model->num_layers]->data[0][0];
      for (int i = 1; i < model->layers[model->num_layers - 1]->output_size;
           i++) {
        if (thread_activations[tid][model->num_layers]->data[i][0] > max_val) {
          max_val = thread_activations[tid][model->num_layers]->data[i][0];
          pred = i;
        }
      }
      if (targets[sample]->data[pred][0] == 1.0)
        correct++;

      backward_pass(model, thread_activations[tid], targets[sample]);

      // ===============
      // loss
      for (int i = 0; i < targets[sample]->rows; i++) {
        if (targets[sample]->data[i][0] == 1.0) {
          epoch_loss +=
              -log(thread_activations[tid][model->num_layers]->data[i][0]);
          break;
        }
      }
    }

    // Print epoch stats
    double epoch_time = omp_get_wtime() - epoch_start;
    printf("Epoch %d/%d | Time: %.2fs | Loss: %.4f | Acc: %.2f%%\n", epoch + 1,
           epochs, epoch_time, epoch_loss / batch_size,
           (double)correct / batch_size * 100);
  }

// Cleanup
#ifdef USE_OMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    for (int i = 0; i <= model->num_layers; i++) {
      matrix_Free(thread_activations[tid][i]);
    }
  }
  printf("Total training time: %.2f seconds\n", omp_get_wtime() - total_start);
}

// =======================
// Activation functions

void relu(struct Matrix *mat) {
  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->columns; j++) {
      mat->data[i][j] = fmax(0.0, mat->data[i][j]);
    }
  }
}

void relu_derivative(struct Matrix *mat) {
  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->columns; j++) {
      mat->data[i][j] = mat->data[i][j] > 0 ? 1.0 : 0.0;
    }
  }
}

void softmax(struct Matrix *mat) {
  double max_val = mat->data[0][0];
  for (int i = 1; i < mat->rows; i++) {
    if (mat->data[i][0] > max_val)
      max_val = mat->data[i][0];
  }

  double sum = 0.0;
  for (int i = 0; i < mat->rows; i++) {
    mat->data[i][0] = exp(mat->data[i][0] - max_val);
    sum += mat->data[i][0];
  }

  for (int i = 0; i < mat->rows; i++) {
    mat->data[i][0] /= sum;
  }
}

// =======================
// Initialization functions

double xavier_scale(int input_size) { return sqrt(2.0 / input_size); }

void xavier_init(struct Matrix *weights) {
  double scale = xavier_scale(weights->columns);
  for (int i = 0; i < weights->rows; i++) {
    for (int j = 0; j < weights->columns; j++) {
      weights->data[i][j] = (2.0 * rand() / RAND_MAX - 1.0) * scale;
    }
  }
}

void random_bias_init(struct Matrix *biases) {
  for (int i = 0; i < biases->rows; i++) {
    biases->data[i][0] = (double)rand() / RAND_MAX * 0.01;
  }
}
