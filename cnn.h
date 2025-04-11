/*
  CNN implementation in C
  similar to pytorch/tensorflow convention
*/

#ifndef CNN_H
#define CNN_H

#include <math.h>
#include <stdlib.h>

typedef struct {
  float *data;
  int rows;
  int columns;
  int channels;
} Matrix_t;

typedef enum {
  CONV_LAYER,
  MAXPOOL_LAYER,
  DENSE_LAYER,
  FLATTER_LAYER,
  // TODO add dropout layer, randomly disable specific neurons for robustness
} LayerType;

typedef enum {
  RELU,
  SOFTMAX
  /* can add others here like sigmoid...*/
} ActivationFunctionType;

/*
  stores the fixed properties of a layer
  as in, the architechture of the layer
*/
typedef struct {
  LayerType type;
  ActivationFunctionType activation;

  int kernel_size;
  int filters;
  int padding;

  int neurons; // for dense layers

  // dropout rate for dropout layers
} LayerConfig_t;

/*
  stores weights and their gradients
  these values are changed during training
*/
typedef struct {

  Matrix_t *weights;
  Matrix_t *biases;
  Matrix_t *weight_grads;
  Matrix_t *bias_grads;

} LayerParams_t;

/*
  temporary working more for propagation
*/
typedef struct {
  Matrix_t *input;
  Matrix_t *output;
  Matrix_t *activated;
  Matrix_t *pool_indicies; // TODO for maxpool backprop
} LayerCache_t;

typedef struct CNNLayer {

  LayerConfig_t config;
  LayerParams_t params;
  LayerCache_t cache;

  struct CNNLayer *next;

} CNNLayer_t;

typedef struct CNNModel {

  int num_layers;
  CNNLayer_t *layers;

  float learning_rate;

  void (*forward)(struct CNNModel *, Matrix_t *);
  void (*backward)(struct CNNModel *, Matrix_t *, float *);
  void (*update)(struct CNNModel *);

} CNNModel_t;

/*
  Core API
  ---
*/

// Init
CNNModel_t *create_Model(float learning_Rate);
void add_Layer(CNNModel_t *model, LayerConfig_t config);
void init_Weights(CNNModel_t *model);

// Propagation
void forward_Pass(CNNModel_t *model, Matrix_t *input);
void backward_Pass(CNNModel_t *model, Matrix_t *target, float *loss);

// Memory management
void free_Model(CNNModel_t *model);
void free_Matrix(Matrix_t *matrix);

// Training
void train_Batch(CNNModel_t *model, Matrix_t *inputs, Matrix_t *targets,
                 int batch_size);
void update_Weights(CNNModel_t *model);

// Evaluation
float calculate_accuracy(CNNModel_t *model, Matrix_t *test_inputs,
                         Matrix_t *test_labels);
Matrix_t *predict(CNNModel_t *model, Matrix_t *input);

// Activation functions
void relu_forward(Matrix_t *mat);
void relu_backward(Matrix_t *mat, Matrix_t *grad);
void softmax_forward(Matrix_t *mat);

// Layer Operations
void conv2d_Forward(CNNLayer_t *layer, Matrix_t *input);
void conv2d_Backward(CNNLayer_t *layer, Matrix_t *grad_output);
void maxpool_Forward(CNNLayer_t *layer, Matrix_t *input);
void maxpool_Backward(CNNLayer_t *layer, Matrix_t *grad_output);
void dense_Forward(CNNLayer_t *layer, Matrix_t *input);
void dense_Backward(CNNLayer_t *layer, Matrix_t *grad_output);

// Utils
Matrix_t *create_Matrix(int rows, int cols, int channels);
Matrix_t *matrix_Copy(Matrix_t *src);
void random_Init(Matrix_t *mat, float min, float max);
void zero_Init(Matrix_t *mat);
void img2col(Matrix_t *input, int output_Height, int output_Width, int channels,
             int kernel_Size, Matrix_t *output); // image to column
void col2img(Matrix_t *cols, int output_Height, int output_Width, int channels,
             int kernel_Size, Matrix_t *output); // column to image

#endif
