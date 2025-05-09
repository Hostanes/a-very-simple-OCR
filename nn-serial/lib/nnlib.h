
#define INPUT_SIZE 784 // nb of floats for each input

void forward_Pass(float *batch, float *weights, float *biases,
                  float *neuron_Values, int batch_Size, int *layer_Sizes,
                  int num_Layers);
