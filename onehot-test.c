#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {

  int label = 2;

  float Activations[] = {0.1, 0.1, 0.8, 0, 0, 0, 0, 0, 0, 0};
  float *error = malloc(sizeof(float) * 10);

  for (int neuron = 0; neuron < 10; neuron++) {
    error[neuron] =
        label == neuron ? Activations[neuron] - 1 : Activations[neuron];
    printf("error value %f\n", error[neuron]);
  }

  return 0;
}
