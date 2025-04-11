#include "cnn.h"
#include <stdio.h>

int main() {
  // Create Model
  CNNModel_t *model = create_Model(0.01f); // Learning rate = 0.01

  // Define Architecture
  LayerConfig_t conv1 = {.type = CONV_LAYER,
                         .activation = RELU,
                         .kernel_size = 3,
                         .filters = 32,
                         .padding = 1};

  LayerConfig_t pool1 = {.type = MAXPOOL_LAYER, .kernel_size = 2};

  LayerConfig_t dense1 = {
      .type = DENSE_LAYER, .activation = RELU, .neurons = 64};

  LayerConfig_t output = {
      .type = DENSE_LAYER, .activation = SOFTMAX, .neurons = 10};

  // Add Layers
  add_Layer(model, conv1);
  add_Layer(model, pool1);
  add_Layer(model, dense1);
  add_Layer(model, output);

  // Initialize Weights
  init_Weights(model);

  // Cleanup
  free_Model(model);

  return 0;
}
