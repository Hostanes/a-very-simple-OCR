#include "nn.h"
#include <omp.h>
#include <time.h>

#define NUM_LAYERS 8

int main() {
  srand(time(NULL));

  const int layer_sizes[NUM_LAYERS + 1] = {8192, 4096, 2048, 1024, 512,
                                           256,  128,  64,   16};

  Model *model = create_model(NUM_LAYERS, layer_sizes, 0.01);

  train_model(model, 10);

  free_model(model);

  return 0;
}
