/*
  Incomplete implementation, currently only includes half forward prop,
  activation functions need more work TODO
*/

#define USE_MNIST_LOADER
#define MNIST_BINARY
#include "lib/matrix-math.h"
#include "mnist-dataloader/mnist-dataloader.h"

#include <math.h>
#include <stdio.h>

#define H1_SIZE 128
#define H2_SIZE 64

void ReLU(Matrix *mat) {
  printf("FUNC: ReLU\n");
  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->columns; j++) {
      mat->data[i][j] = fmax(0.0, mat->data[i][j]);
    }
  }
}

void softmax(Matrix *mat) {

  printf("Softmax\n");
  double sum = 0.0;
  for (int i = 0; i < mat->rows; i++) {
    mat->data[i][0] = exp(mat->data[i][0]);
    sum += mat->data[i][0];
  }
  for (int i = 0; i < mat->rows; i++) {
    mat->data[i][0] /= sum;
  }
}

int forward_Prop_Layer(Matrix *input, Matrix *output, Matrix *weights,
                       void (*activationFunc)(Matrix *mat)) {
  printf("Forward layers w: %d, %d, i: %d, %d a: %d, %d\n", weights->rows,
         weights->columns, input->rows, input->columns, output->rows,
         output->columns);
  Matrix *temp = dot_Mat(weights, input);
  activationFunc(output);

  for (int i = 0; i < output->rows; i++) {
    output->data[i][0] = temp->data[i][0];
  }

  return 0;
}

int init_Layers() {
  //

  return 0;
}

int main(int argc, char **argv) {
  mnist_data *data;
  unsigned int cnt;
  int ret;

  if (ret = mnist_load("dataset/train-images.idx3-ubyte",
                       "dataset/train-labels.idx1-ubyte", &data, &cnt)) {
    printf("An error occured: %d\n", ret);
  } else {
    printf("image count: %d\n", cnt);
  }

  /*
    Pipeline:
      - Input to H1: ReLU
      - H1 to H2: ReLU
      - H2 to Output: Softmax

      TODO
      double check if these are the activation functions we need, end result
    should be from 0 to 1 for each neuron, and sum of all = 1
        */

  Matrix *w1 = init_Matrix(H1_SIZE, 784);
  Matrix *w2 = init_Matrix(H2_SIZE, H1_SIZE);
  Matrix *w3 = init_Matrix(10, H2_SIZE);

  randomize_Matrix(w1);
  randomize_Matrix(w2);
  randomize_Matrix(w3);

  print_Matrix(w3);

  Matrix *a1 = init_Matrix(H1_SIZE, 1);
  Matrix *a2 = init_Matrix(H2_SIZE, 1);
  Matrix *output = init_Matrix(10, 1);

  Matrix *input = init_Matrix(784, 1);
  for (int i = 0; i < 784; i++) {
    input->data[i][0] = data[0].data[i];
  }

  forward_Prop_Layer(input, a1, w1, ReLU);
  forward_Prop_Layer(a1, a2, w2, ReLU);
  forward_Prop_Layer(a2, output, w3, softmax);

  print_Matrix(output);

  free(data);

  return 0;
}
