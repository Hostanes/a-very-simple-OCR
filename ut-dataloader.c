/*
  ut-dataloader.c

  unit test of the mnist data loader, prints out a random image in the terminal
  along with its label
*/


#define USE_MNIST_LOADER
#define MNIST_BINARY
#include "mnist-dataloader/mnist-dataloader.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

  srand(time(0));

  int index = rand() % (60001);
  mnist_data image = data[index];

  printf("displaying image of index %d:\n", index);
  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      printf("%c", image.data[i * 28 + j] == 1 ? '#' : '.');
    }
    printf("\n");
  }
  printf("label: %d\n", image.label);

  free(data);
  return 0;
}
