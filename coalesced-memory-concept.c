#include <stdio.h>


int main() {

  int number_Of_Inputs = 10;
  int size_Of_Input = 784;
  float *input;

  int number_Of_Layers = 5;
  int layer_Sizes[] = {784, 512, 256, 128, 10};

  // stores weights of all layers in a single flat array of this size
  // [Weights1 Row 1, Weights1 Row 2, ...  Weights 1 Row 512, Weights 2 Row1,
  // .....]
  int number_Of_Weight_Floats = layer_Sizes[0] * layer_Sizes[1];

  for (int i = 1; i < number_Of_Layers - 1; i++) {
    number_Of_Weight_Floats += layer_Sizes[i] * layer_Sizes[i + 1];
  }

  // weights 1d array, stored Row of weights next to other row of weights
  float Weights[number_Of_Weight_Floats];

  printf("total number of weight floats = %d\n", number_Of_Weight_Floats);

  /* TEST Example on how to access the weights using this flattening method
  {
    // How to access a weight
    // Example:
    // Weight 2 row 80 element 400
    int row = 80;     // out of 256
    int column = 400; // out of 512
    int index =
        (layer_Sizes[0] * layer_Sizes[1]) + (layer_Sizes[1] * row) + column;
   printf("example index at weight 2 row 80 element 400 = %d\n", index);
    // preferably we provide the start index to each
    // weight matrix to the kernel
    // to ease calculations
  }
  */

  // Similar thing for biases

  int number_of_bias_floats = 0;

  for (int i = 1; i < number_Of_Layers; i++) {
    number_of_bias_floats += layer_Sizes[i];
  }

  float Biases[number_of_bias_floats];

  /* TEST
  {
    // Example: Accessing bias for layer 2, neuron 80
    // layer 2 is index 2 (0-based), so it's the 2nd bias block (after biases
    // for layer 1)
    int bias_index = 0;
    for (int i = 1; i < 2; i++) {
      bias_index += layer_Sizes[i];
    }
    bias_index += 80; // neuron index within layer 2
    printf("example index at bias for layer 2, neuron 80 = %d\n", bias_index);
  }
  */

  /*
    in place activation and Z value buffers for each layer
    we write on these for each batch

    Stored as:
    Z 1 1, A1 1
    Z 1 2, A1 2 (layer 1 neuron 2)
    ...
    Z 1 512, A 1 512
    Z 2 1, A 2 1 (start of new layer)
  */

  int neuron_Values_Sizes = 0;

  for (int i = 1; i < number_Of_Layers; i++) {
    neuron_Values_Sizes += layer_Sizes[i] * 2;
  }
  float neuron_Values[neuron_Values_Sizes];

  return 0;
}
