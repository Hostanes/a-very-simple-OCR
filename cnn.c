#include "cnn.h"
#include <float.h>
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
/*
  applies convolution and defaults to ReLU at the end
  TODO add multiple activation function support
*/
void conv2d_Forward(CNNLayer_t *layer, Matrix_t *input) {

  if (!layer || !input || !layer->params.weights || !layer->params.biases) {
    fprintf(stderr, "Error: Null parameters in conv2d_Forward\n");
    return;
  }
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

/*
  Basic 2x2 max pooling
  TODO actually understand how this works, I just copy pasted it
*/
void maxpool_Forward(CNNLayer_t *layer, Matrix_t *input) {
  const int pool_size = 2;
  const int stride = 2;

  // Calculate output dimensions
  const int out_h = input->rows / pool_size;
  const int out_w = input->columns / pool_size;

  // Allocate output and indices matrices
  layer->cache.output = create_Matrix(out_h, out_w, input->channels);
  layer->cache.pool_indicies = create_Matrix(
      out_h, out_w, input->channels * 2); // Stores (h,w) positions

  for (int c = 0; c < input->channels; c++) {
    for (int h = 0; h < out_h; h++) {
      for (int w = 0; w < out_w; w++) {
        float max_val = -FLT_MAX;
        int max_h = 0, max_w = 0;

        // Find maximum in pooling window
        for (int ph = 0; ph < pool_size; ph++) {
          for (int pw = 0; pw < pool_size; pw++) {
            const int h_in = h * stride + ph;
            const int w_in = w * stride + pw;

            if (h_in < input->rows && w_in < input->columns) {
              const int idx =
                  (h_in * input->columns + w_in) * input->channels + c;
              if (input->data[idx] > max_val) {
                max_val = input->data[idx];
                max_h = h_in;
                max_w = w_in;
              }
            }
          }
        }

        // Store results
        const int out_idx = (h * out_w + w) * input->channels + c;
        layer->cache.output->data[out_idx] = max_val;

        // Store indices for backprop
        layer->cache.pool_indicies->data[out_idx * 2] = max_h;
        layer->cache.pool_indicies->data[out_idx * 2 + 1] = max_w;
      }
    }
  }
}

/*
  basic dense (fully connected) neural network function
  1. w.x + b = c
  2. output = act(c)

  w: weight matrix
  x: input vector
  b: bias
  act: activation function

  TODO add support for more channels.
  DNN might only take 1d vectors, so maybe current
  approach for flattening multi dimensional mats is correct?
*/
void dense_Forward(CNNLayer_t *layer, Matrix_t *input) {
  int input_size = input->columns; // assuming input shape is (1, input_size, 1)
  int output_size =
      layer->params.weights->rows; // shape: (output_size, input_size)

  layer->cache.output = create_Matrix(1, output_size, 1);

  for (int o = 0; o < output_size; o++) {
    float sum = 0.0f;
    for (int i = 0; i < input_size; i++) {
      int weight_idx = o * input_size + i;
      sum += layer->params.weights->data[weight_idx] * input->data[i];
    }
    sum += layer->params.biases->data[o];
    layer->cache.output->data[o] = sum;
  }

  // Activation
  layer->cache.activated = matrix_Copy(layer->cache.output);

  if (layer->config.activation == RELU) {
    relu_forward(layer->cache.activated);
  } else if (layer->config.activation == SOFTMAX) {
    softmax_forward(layer->cache.output, layer->cache.activated);
  }
}

/*
  applies ReLU to an entire matrix
  output = max(0,x_i)
*/
void relu_forward(Matrix_t *mat) {
  // Check for null pointer
  if (!mat || !mat->data) {
    fprintf(stderr, "Error: Null matrix in relu_forward\n");
    return;
  }

  // Get total number of elements in matrix
  const int total_elements = mat->rows * mat->columns * mat->channels;

  // Apply ReLU to each element
  for (int i = 0; i < total_elements; i++) {
    mat->data[i] = mat->data[i] > 0 ? mat->data[i] : 0;
  }
}

/*
  Applies softmax to input and placed into output matrix
*/
void softmax_forward(Matrix_t *input, Matrix_t *output) {

  float max_val = -FLT_MAX;
  float sum_exp = 0.0f;
  const int size = input->rows * input->columns;

  // Find max value for numerical stability
  for (int i = 0; i < size; i++) {
    if (input->data[i] > max_val) {
      max_val = input->data[i];
    }
  }

  // Compute exponentials and their sum
  for (int i = 0; i < size; i++) {
    output->data[i] = expf(input->data[i] - max_val);
    sum_exp += output->data[i];
  }

  // Normalize to get probabilities
  for (int i = 0; i < size; i++) {
    output->data[i] /= sum_exp;
  }
}

/*
  Flatten a multidimensional matrix
  supports flattening multiple channels too!!
*/
void flatten_Forward(CNNLayer_t *layer, Matrix_t *input) {
  int flat_size = input->rows * input->columns * input->channels;

  // Allocate flat output: 1 row, flat_size columns, 1 channel
  layer->cache.output = create_Matrix(1, flat_size, 1);

  for (int i = 0; i < flat_size; i++) {
    layer->cache.output->data[i] = input->data[i];
  }

  // For simplicity, activated == output here
  layer->cache.activated = layer->cache.output;
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
  // Get layer parameters
  int padding = layer->config.padding;
  int kernel_size = layer->config.kernel_size;
  Matrix_t *input = layer->cache.input;

  // Initialize gradients if they don't exist
  if (layer->params.weight_grads == NULL) {
    layer->params.weight_grads = matrix_Copy(layer->params.weights);
    zero_Init(layer->params.weight_grads);
  }
  if (layer->params.bias_grads == NULL) {
    layer->params.bias_grads = matrix_Copy(layer->params.biases);
    zero_Init(layer->params.bias_grads);
  }

  // Compute gradient of ReLU activation
  Matrix_t *grad_act = matrix_Copy(grad_output);
  for (int i = 0; i < grad_act->rows * grad_act->columns * grad_act->channels;
       i++) {
    if (layer->cache.output->data[i] <= 0) {
      grad_act->data[i] = 0;
    }
  }

  // Compute input gradient (∂L/∂X)
  Matrix_t *grad_input =
      create_Matrix(input->rows, input->columns, input->channels);
  zero_Init(grad_input);

  // Compute weight gradients (∂L/∂W) and bias gradients (∂L/∂B)
  for (int f = 0; f < layer->config.filters; f++) {
    for (int h = 0; h < grad_output->rows; h++) {
      for (int w = 0; w < grad_output->columns; w++) {
        float grad =
            grad_act
                ->data[(h * grad_output->columns + w) * grad_output->channels +
                       f];

        // Update bias gradient
        layer->params.bias_grads->data[f] += grad;

        // Compute weight gradients
        for (int kh = 0; kh < kernel_size; kh++) {
          for (int kw = 0; kw < kernel_size; kw++) {
            for (int ch = 0; ch < input->channels; ch++) {
              int h_in = h + kh - padding;
              int w_in = w + kw - padding;

              if (h_in >= 0 && h_in < input->rows && w_in >= 0 &&
                  w_in < input->columns) {
                int weight_idx =
                    ((kh * kernel_size + kw) * input->channels + ch) *
                        layer->config.filters +
                    f;
                int input_idx =
                    (h_in * input->columns + w_in) * input->channels + ch;

                // Accumulate weight gradient
                layer->params.weight_grads->data[weight_idx] +=
                    grad * input->data[input_idx];

                // Accumulate input gradient
                grad_input->data[input_idx] +=
                    grad * layer->params.weights->data[weight_idx];
              }
            }
          }
        }
      }
    }
  }

  // Store input gradient for next layer
  layer->cache.input = grad_input;
  free_Matrix(grad_act);
}

void maxpool_Backward(CNNLayer_t *layer, Matrix_t *grad_output) {
  Matrix_t *input = layer->cache.input;
  Matrix_t *grad_input =
      create_Matrix(input->rows, input->columns, input->channels);
  zero_Init(grad_input);

  const int pool_size = 2;

  for (int c = 0; c < grad_output->channels; c++) {
    for (int h = 0; h < grad_output->rows; h++) {
      for (int w = 0; w < grad_output->columns; w++) {
        // Get stored max positions
        int out_idx =
            (h * grad_output->columns + w) * grad_output->channels + c;
        int max_h = layer->cache.pool_indicies->data[out_idx * 2];
        int max_w = layer->cache.pool_indicies->data[out_idx * 2 + 1];

        // Route gradient to max position
        int in_idx = (max_h * input->columns + max_w) * input->channels + c;
        grad_input->data[in_idx] = grad_output->data[out_idx];
      }
    }
  }

  layer->cache.input = grad_input;
}

void dense_Backward(CNNLayer_t *layer, Matrix_t *grad_output) {
  Matrix_t *input = layer->cache.input;
  int input_size = input->columns;
  int output_size = layer->params.weights->rows;

  // Initialize gradients if they don't exist
  if (layer->params.weight_grads == NULL) {
    layer->params.weight_grads = matrix_Copy(layer->params.weights);
    zero_Init(layer->params.weight_grads);
  }
  if (layer->params.bias_grads == NULL) {
    layer->params.bias_grads = matrix_Copy(layer->params.biases);
    zero_Init(layer->params.bias_grads);
  }

  // Compute gradient of activation (ReLU)
  Matrix_t *grad_act = matrix_Copy(grad_output);
  if (layer->config.activation == RELU) {
    for (int i = 0; i < grad_act->rows * grad_act->columns * grad_act->channels;
         i++) {
      if (layer->cache.output->data[i] <= 0) {
        grad_act->data[i] = 0;
      }
    }
  }

  // Compute weight gradients (∂L/∂W)
  for (int i = 0; i < input_size; i++) {
    for (int o = 0; o < output_size; o++) {
      int weight_idx = o * input_size + i;
      layer->params.weight_grads->data[weight_idx] +=
          grad_act->data[o] * input->data[i];
    }
  }

  // Compute bias gradients (∂L/∂B)
  for (int o = 0; o < output_size; o++) {
    layer->params.bias_grads->data[o] += grad_act->data[o];
  }

  // Compute input gradients (∂L/∂X)
  Matrix_t *grad_input = create_Matrix(1, input_size, 1);
  zero_Init(grad_input);

  for (int i = 0; i < input_size; i++) {
    for (int o = 0; o < output_size; o++) {
      int weight_idx = o * input_size + i;
      grad_input->data[i] +=
          grad_act->data[o] * layer->params.weights->data[weight_idx];
    }
  }

  layer->cache.input = grad_input;
  free_Matrix(grad_act);
}

void update_Weights(CNNModel_t *model) {
  CNNLayer_t *layer = model->layers;
  while (layer != NULL) {
    if (layer->params.weight_grads != NULL &&
        layer->params.bias_grads != NULL) {
      // Update weights
      for (int i = 0;
           i < layer->params.weights->rows * layer->params.weights->columns *
                   layer->params.weights->channels;
           i++) {
        layer->params.weights->data[i] -=
            model->learning_rate * layer->params.weight_grads->data[i];
      }

      // Update biases
      for (int i = 0;
           i < layer->params.biases->rows * layer->params.biases->columns *
                   layer->params.biases->channels;
           i++) {
        layer->params.biases->data[i] -=
            model->learning_rate * layer->params.bias_grads->data[i];
      }

      // Reset gradients
      zero_Init(layer->params.weight_grads);
      zero_Init(layer->params.bias_grads);
    }
    layer = layer->next;
  }
}

Matrix_t *cross_Entropy_Backward(Matrix_t *predicted, int true_class_index) {
  Matrix_t *grad =
      create_Matrix(predicted->rows, predicted->columns, predicted->channels);
  for (int i = 0; i < predicted->columns; ++i) {
    grad->data[i] = predicted->data[i]; // copy prediction
  }
  grad->data[true_class_index] -= 1.0f; // subtract 1 from correct class
  return grad;
}

float cross_Entropy_Loss(Matrix_t *predicted, int true_class_index) {
  float epsilon = 1e-7f; // Avoid log(0)
  float predicted_prob = predicted->data[true_class_index];
  if (predicted_prob < epsilon)
    predicted_prob = epsilon;
  return -logf(predicted_prob);
}

/*
  Weight managment
*/

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
  int input_channels = 1;

  while (layer != NULL) {
    switch (layer->config.type) {

    case CONV_LAYER: {
      // Fan_in = kernel_size^2 * input_channels
      // Fan_out = kernel_size^2 * filters
      int fan_in = layer->config.kernel_size * layer->config.kernel_size *
                   input_channels;
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
