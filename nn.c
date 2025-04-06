#define USE_MNIST_LOADER
#define MNIST_BINARY
#include "lib/matrix-math.h"
#include "mnist-dataloader/mnist-dataloader.h"

#include <stdio.h>

#define H1_SIZE 128
#define H2_SIZE 64

int forward_Propagation() {
  //

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
    Matricies needed:
    - W1 784 x 128 (input -> H1)
    - W2 128 x 64 (H1 -> H2)
    - W1 64 x 10 (H2 -> output)

    use HE init to randomize initial weights

    - b1 128 x 1
    - b2 64 x 1
    - b3 10 x1

    iniitialize to 0

    Pipeline:

    Forward prop
    - I -> H1 (ReLU)
      - compute Z1 = dot(W1, input) + b1 (128 x 1 weighted sums of H1)
      - activation A1= ReLU(Z1) (output values of H1)
    - H1 -> H2 (ReLU)
      - Z2 = dot(W2, A1) + b2 (64 x 1)
      - A2 = ReLU(Z2)
    - H2 -> output (SoftMax)
      - Z3 = dot(W3, A2) (10 x 1)
      - A3 = softmax(Z3)

    Loss calulcated with cross-entropy

    Backward prop

  */

  free(data);
  return 0;
}
