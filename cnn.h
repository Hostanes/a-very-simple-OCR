
#ifndef CNN_H
#define CNN_H

#include <stdlib.h>

typedef struct {
  float *data;
  int rows;
  int column;
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

typedef struct {
  LayerType type;
  ActivationFunctionType activation;

  int kernel_size;
  int filters;
  int padding;

  int neurons; // for dense layers

  // dropout rate for dropout layers
} LayerConfig_t;

typedef struct {

  Matrix_t *input;
  Matrix_t *output;
  Matrix_t *weight_grads;
  Matrix_t *bias_grads;

} LayerParams_t;

typedef struct {
  Matrix_t *input;
  Matrix_t *output;
  Matrix_t *activated;
  Matrix_t *pool_indicies; // TODO for maxpool backprop
} LayerCache_t;

typedef struct CNNLayer_t {

  LayerConfig_t config;
  LayerParams_t params;
  LayerCache_t cache;

  struct CNNLayer_t *next;

} CNNLayer_t;

typedef struct CNNModel_t {

  int num_layers;
  CNNLayer_t *layers;

  float learning_rate;
  float momentum; // TODO
  float decay;    // TODO

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
void add_Layer(CNNLayer_t *model, LayerConfig_t config);
void init_Weights(CNNModel_t *model);

// Propagation
void forward_Pass(CNNModel *model, Matrix_t *input);
void backward_Pass(CNNModel_t *model, Matrix_t *target, float *loss);

// Training
void free_Model(CNNModel_t *model);
void free_Matrix(Matrix_t *matrix);

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

#endif
