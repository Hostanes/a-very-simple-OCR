#include "cnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// function implementations

CNNModel_t *create_Model(float learning_Rate) {
  CNNModel_t *model = (CNNModel_t *)malloc(sizeof(CNNModel_t));

  model->num_layers = 0;
  model->layers = NULL;
  model->learning_rate = learning_Rate;

  model->forward = forward_Pass;
  model->backward = backward_Pass;
  model->update = update_Weights;

  return model;
}

// Adds a layer to the model
void add_Layer(CNNModel_t *model, LayerConfig_t config) {
  CNNLayer_t *new_layer = (CNNLayer_t *)malloc(sizeof(CNNLayer_t));
  new_layer->config = config;
  new_layer->next = NULL;

  new_layer->params.weight_grads = new_layer->params.bias_grads = NULL;
  new_layer->cache.input = new_layer->cache.output = NULL;
  new_layer->cache.activated = new_layer->cache.pool_indicies = NULL;

  /*
    layers operate in a linked list fashion,
    with each layer storing pointer to next layer
  */
  if (model->layers == NULL) {
    model->layers = new_layer;
  } else {
    CNNLayer_t *current = model->layers;
    while (current->next != NULL) {
      current = current->next;
    }
    current->next = new_layer;
  }
  model->num_layers++;
}

void update_Weights(CNNModel_t *model) {
  //
  //
}

/*
  Propagation
*/

// Forward pass through the network
void forward_Pass(CNNModel_t *model, Matrix_t *input) {
  CNNLayer_t *current = model->layers;
  Matrix_t *layer_input = input;

  while (current != NULL) {
    current->cache.input = layer_input;

    switch (current->config.type) {
    case CONV_LAYER:
      conv2d_Forward(current, layer_input);
      layer_input = current->cache.activated;
      break;
    case MAXPOOL_LAYER:
      maxpool_Forward(current, layer_input);
      layer_input = current->cache.output;
      break;
    case DENSE_LAYER:
      dense_Forward(current, layer_input);
      layer_input = current->cache.activated;
      break;
    default:
      break;
    }

    current = current->next;
  }
}

void conv2d_Forward(CNNLayer_t *layer, Matrix_t *input) {
  //

  int padding = layer->config.padding;
  int kernel_size = layer->config.kernel_size;

  int output_height = (input->rows + 2 * padding - kernel_size) + 1;
  int output_width = (input->columns + 2 * padding - kernel_size) + 1;

  layer->cache.output =
      create_Matrix(output_height, output_width, layer->config.filters);

  for (int i_filter = 0; i_filter < layer->config.filters; i_filter++) {

    for (int h = 0; h < output_height; h++) {
      for (int w = 0; w < output_width; w++) {
        float sum = layer->params.biases
                        ->data[i_filter]; // add bias for i_filter to sum

        int h_start = h - padding;
        int w_start = w - padding;

        for (int kh = 0; kh < kernel_size; kh++) {         // kernel height
          for (int kw = 0; kw < kernel_size; kw++) {       // kernel width
            for (int ch = 0; ch < input->channels; ch++) { // input channels

              int h_in = h_start + kh;
              int w_in = w_start + kw;

              if (h_in < 0 || h_in >= input->rows || w_in < 0 ||
                  w_in >= input->columns) {
                continue;
              }
              /*
                calculate weight index for the current filter, conceptually 4
                dimensions but stored as 1 dimension
              */
              int weight_idx =
                  ((kh * kernel_size + kw) * input->channels + ch) *
                      layer->config.filters +
                  i_filter;
              /*
                caclulate index for required input value
              */
              int input_idx =
                  (h_in * input->columns + w_in) * input->channels + ch;

              sum += layer->params.weights->data[weight_idx] *
                     input->data[input_idx];
            }
          }
        }
        int out_idx = (h * output_width + w) * layer->config.filters + i_filter;
        layer->cache.output->data[out_idx] = sum;
      }
    }
  }

  // Apply activation (ReLU)
  layer->cache.activated = matrix_Copy(layer->cache.output);
  relu_forward(layer->cache.activated);
}

void maxpool_Forward(CNNLayer_t *layer, Matrix_t *input) {
  //
  //
}

void dense_Forward(CNNLayer_t *layer, Matrix_t *input) {
  //
  //
}

void relu_forward(Matrix_t *mat) {
  //
  //
}

void softmax_forward(Matrix_t *mat) {
  //
  //
}

// Backward pass through the network
void backward_Pass(CNNModel_t *model, Matrix_t *target, float *loss) {
  // Calculate initial gradient (softmax + cross-entropy)
  CNNLayer_t *output_layer = model->layers;
  while (output_layer->next != NULL)
    output_layer = output_layer->next;

  *loss = 0.0f;
  for (int i = 0; i < target->rows * target->columns * target->channels; i++) {
    *loss += -target->data[i] * log(output_layer->cache.output->data[i]);
    output_layer->cache.output->data[i] =
        output_layer->cache.output->data[i] - target->data[i];
  }

  // Backpropagate through layers
  CNNLayer_t *current = output_layer;
  Matrix_t *grad_output = output_layer->cache.output;

  while (current != NULL) {
    switch (current->config.type) {
    case CONV_LAYER:
      conv2d_Backward(current, grad_output);
      grad_output = current->cache.input;
      break;
    case MAXPOOL_LAYER:
      maxpool_Backward(current, grad_output);
      grad_output = current->cache.input;
      break;
    case DENSE_LAYER:
      dense_Backward(current, grad_output);
      grad_output = current->cache.input;
      break;
    default:
      break;
    }

    current = current->next; // Move backward through the network
  }
}

void conv2d_Backward(CNNLayer_t *layer, Matrix_t *grad_output) {
  //
  //
}

void maxpool_Backward(CNNLayer_t *layer, Matrix_t *grad_output) {
  //
  //
}

void dense_Backward(CNNLayer_t *layer, Matrix_t *grad_output) {
  //
  //
}

/*
  Using HE init
*/
void init_Conv_Weights(Matrix_t *weights, int fan_in, int fan_out) {
  float stddev = sqrt(2.0f / (fan_in + fan_out));
  random_Init(weights, -stddev, stddev);
}

/*
  Using Glorot init
*/
void init_Dense_Weights(Matrix_t *weights, int fan_in, int fan_out) {
  float limit = sqrt(6.0f / (fan_in + fan_out));
  random_Init(weights, -limit, limit);
}

/*
  Initialize all Matrix values to 0
*/
void zero_Init(Matrix_t *mat) {
  for (int i = 0; i < mat->rows * mat->columns * mat->channels; i++) {
    mat->data[i] = 0.0f;
  }
}

/*
  DEPRECIATED
  Using random init
  used for testing
*/
void random_Init(Matrix_t *mat, float min, float max) {
  float range = max - min;
  for (int i = 0; i < mat->rows * mat->columns * mat->channels; i++) {
    mat->data[i] = min + (rand() / (float)RAND_MAX) * range;
  }
}

void init_Weights(CNNModel_t *model) {
  CNNLayer_t *layer = model->layers;
  while (layer != NULL) {
    switch (layer->config.type) {

    case CONV_LAYER: {
      // Fan_in = kernel_size^2 * input_channels
      // Fan_out = kernel_size^2 * filters
      int fan_in = layer->config.kernel_size * layer->config.kernel_size *
                   (layer == model->layers ? 1 : layer->cache.input->channels);
      int fan_out = layer->config.kernel_size * layer->config.kernel_size *
                    layer->config.filters;

      layer->params.weights =
          create_Matrix(layer->config.kernel_size, layer->config.kernel_size,
                        fan_in * layer->config.filters);
      init_Conv_Weights(layer->params.weights, fan_in, fan_out);

      layer->params.biases = create_Matrix(1, 1, layer->config.filters);
      zero_Init(layer->params.biases);
      break;
    }

    case DENSE_LAYER: {
      int input_size = layer->cache.input->rows * layer->cache.input->columns *
                       layer->cache.input->channels;
      int output_size = layer->config.neurons;

      layer->params.weights = create_Matrix(input_size, output_size, 1);
      init_Dense_Weights(layer->params.weights, input_size, output_size);

      layer->params.biases = create_Matrix(1, output_size, 1);
      zero_Init(layer->params.biases);
      break;
    }

    default: {

      break;
    }
    }
    layer = layer->next;
  }
}

/*
  Creates a new matrix with specified dimensions
*/
Matrix_t *create_Matrix(int rows, int cols, int channels) {
  Matrix_t *mat = (Matrix_t *)malloc(sizeof(Matrix_t));
  if (!mat) {
    fprintf(stderr, "Error: Failed to allocate matrix struct\n");
    return NULL;
  }

  mat->rows = rows;
  mat->columns = cols;
  mat->channels = channels;

  size_t num_elements = rows * cols * channels;
  mat->data = (float *)malloc(num_elements * sizeof(float));
  if (!mat->data) {
    fprintf(stderr, "Error: Failed to allocate matrix data\n");
    free(mat);
    return NULL;
  }

  memset(mat->data, 0, num_elements * sizeof(float));

  return mat;
}

/*
  Create a deep copy of a matrix
  copied from src matrix
  returns pointer to new matrix
*/
Matrix_t *matrix_Copy(Matrix_t *src) {
  if (!src) {
    fprintf(stderr, "Error: Cannot copy NULL matrix\n");
    return NULL;
  }

  Matrix_t *dst = create_Matrix(src->rows, src->columns, src->channels);
  if (!dst)
    return NULL;

  size_t num_bytes = src->rows * src->columns * src->channels * sizeof(float);
  memcpy(dst->data, src->data, num_bytes);

  return dst;
}

// Frees a matrix
void free_Matrix(Matrix_t *matrix) {
  if (matrix != NULL) {
    if (matrix->data != NULL) {
      free(matrix->data);
    }
    free(matrix);
  }
}

// Frees the entire model
void free_Model(CNNModel_t *model) {
  CNNLayer_t *current = model->layers;
  while (current != NULL) {
    CNNLayer_t *next = current->next;

    free_Matrix(current->params.weights);
    free_Matrix(current->params.biases);
    free_Matrix(current->params.weight_grads);
    free_Matrix(current->params.bias_grads);

    free_Matrix(current->cache.input);
    free_Matrix(current->cache.output);
    free_Matrix(current->cache.activated);
    free_Matrix(current->cache.pool_indicies);

    free(current);
    current = next;
  }
  free(model);
}
