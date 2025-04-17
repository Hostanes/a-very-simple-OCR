/*
  version of nnlib.c that parallelizes each sample's operations
  parallel:
  - forward
  - backward
  - NOT softmax, mish me7erze
*/

#include "nnlib.h"

float relu(float x) { return x > 0 ? x : 0; }

float relu_derivative(float x) { return x > 0 ? 1 : 0; }

float tanh_activation(float x) { return tanhf(x); }

float tanh_derivative(float x) {
  float t = tanhf(x);
  return 1 - t * t;
}

float linear(float x) { return x; }

float linear_derivative(float x) { return 1; }

void softmax(float *array, int size) {
  // max value
  float max_val = array[0];
  for (int i = 1; i < size; i++) {
    if (array[i] > max_val) {
      max_val = array[i];
    }
  }

  // calculate sum: e^{xi - max}
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    array[i] = expf(array[i] - max_val); // Subtract max for numerical stability
    sum += array[i];
  }

  // normalize
  for (int i = 0; i < size; i++) {
    array[i] /= sum;
  }
}

void softmax_derivative(float *output, float *gradient, int size) {
  for (int i = 0; i < size; i++) {
    gradient[i] = 0;
    for (int j = 0; j < size; j++) {
      float delta = (i == j) ? 1 : 0;
      gradient[i] += output[i] * (delta - output[j]) * gradient[j];
    }
  }
}

void initialize_layer(Layer_t *layer, int input_size, int output_size,
                      ActivationFunc activation,
                      ActivationDerivative derivative) {
  layer->input_size = input_size;
  layer->output_size = output_size;
  layer->activation = activation;
  layer->activation_derivative = derivative;

  int weights_size = input_size * output_size;

  layer->weights = (float *)malloc(weights_size * sizeof(float));
  layer->biases = (float *)calloc(output_size, sizeof(float));
  layer->weight_momentum = (float *)calloc(weights_size, sizeof(float));
  layer->bias_momentum = (float *)calloc(output_size, sizeof(float));
  layer->output = (float *)calloc(output_size, sizeof(float));
  layer->input = (float *)calloc(output_size, sizeof(float));

  // Xavier init
  float scale = sqrtf(2.0f / input_size);
  randomize_weights(layer->weights, weights_size, scale);
}

void randomize_weights(float *weights, int size, float scale) {
  for (int i = 0; i < size; i++) {
    weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
  }
}

NeuralNetwork_t *create_network(int *layer_sizes, int num_layers,
                                ActivationFunc *activations,
                                ActivationDerivative *derivatives,
                                float learning_rate, float momentum) {
  NeuralNetwork_t *net = (NeuralNetwork_t *)malloc(sizeof(NeuralNetwork_t));
  net->num_layers = num_layers - 1; // First size is input layer
  net->layers = (Layer_t *)malloc(net->num_layers * sizeof(Layer_t));
  net->learning_rate = learning_rate;
  net->momentum = momentum;

  for (int i = 0; i < net->num_layers; i++) {
    initialize_layer(&net->layers[i], layer_sizes[i], layer_sizes[i + 1],
                     activations[i], derivatives[i]);
  }

  return net;
}

void free_network(NeuralNetwork_t *net) {
  for (int i = 0; i < net->num_layers; i++) {
    free(net->layers[i].weights);
    free(net->layers[i].biases);
    free(net->layers[i].weight_momentum);
    free(net->layers[i].bias_momentum);
    free(net->layers[i].output);
    free(net->layers[i].input);
  }
  free(net->layers);
  free(net);
}

float *forward_pass(NeuralNetwork_t *net, float *input) {
  float *current_input = input;

  for (int i = 0; i < net->num_layers; i++) {
    Layer_t *layer = &net->layers[i];

// Parallelize the dot product
#pragma omp parallel for
    for (int j = 0; j < layer->output_size; j++) {
      float sum = layer->biases[j];
      for (int k = 0; k < layer->input_size; k++) {
        sum += current_input[k] * layer->weights[k * layer->output_size + j];
      }
      layer->input[j] = sum;
    }

    // Parallelize activation
    if (layer->activation == softmax_placeholder) {
      memcpy(layer->output, layer->input, layer->output_size * sizeof(float));
      softmax(layer->output, layer->output_size);
    } else {
#pragma omp parallel for
      for (int j = 0; j < layer->output_size; j++) {
        layer->output[j] = layer->activation(layer->input[j]);
      }
    }

    current_input = layer->output;
  }
  return current_input;
}

void backward_pass(NeuralNetwork_t *net, float *input, float *target) {
  float *output = forward_pass(net, input);
  Layer_t *output_layer = &net->layers[net->num_layers - 1];

  // Allocate deltas
  float **deltas = malloc(net->num_layers * sizeof(float *));
  for (int i = 0; i < net->num_layers; i++) {
    deltas[i] = calloc(net->layers[i].output_size, sizeof(float));
  }

  // PARALLEL
  if (output_layer->activation == softmax_placeholder) {
#pragma omp parallel for
    for (int i = 0; i < output_layer->output_size; i++) {
      deltas[net->num_layers - 1][i] = output_layer->output[i] - target[i];
    }
  } else {
#pragma omp parallel for
    for (int i = 0; i < output_layer->output_size; i++) {
      float error = output_layer->output[i] - target[i];
      deltas[net->num_layers - 1][i] =
          error * output_layer->activation_derivative(output_layer->input[i]);
    }
  }

  // PARALLEL
  for (int l = net->num_layers - 2; l >= 0; l--) {
    Layer_t *current = &net->layers[l];
    Layer_t *next = &net->layers[l + 1];

#pragma omp parallel for
    for (int i = 0; i < current->output_size; i++) {
      float error = 0;
      for (int j = 0; j < next->output_size; j++) {
        error += next->weights[i * next->output_size + j] * deltas[l + 1][j];
      }
      deltas[l][i] = error * current->activation_derivative(current->input[i]);
    }
  }

  float *prev_output = input;
  for (int l = 0; l < net->num_layers; l++) {
    Layer_t *layer = &net->layers[l];

// PARALLEL
#pragma omp parallel for
    for (int i = 0; i < layer->input_size; i++) {
      for (int j = 0; j < layer->output_size; j++) {
        int idx = i * layer->output_size + j;
        float gradient = prev_output[i] * deltas[l][j];
        layer->weight_momentum[idx] =
            net->momentum * layer->weight_momentum[idx] +
            net->learning_rate * gradient;
        layer->weights[idx] -= layer->weight_momentum[idx];
      }
    }

// PARALLEL
#pragma omp parallel for
    for (int j = 0; j < layer->output_size; j++) {
      layer->bias_momentum[j] = net->momentum * layer->bias_momentum[j] +
                                net->learning_rate * deltas[l][j];
      layer->biases[j] -= layer->bias_momentum[j];
    }

    prev_output = layer->output;
  }

  // Clean up
  for (int i = 0; i < net->num_layers; i++) {
    free(deltas[i]);
  }
  free(deltas);
}

void train(NeuralNetwork_t *net, float *input, float *target) {
  backward_pass(net, input, target); // backward pass includes forward pass
                                     // function call, TODO fix this
}

int predict(NeuralNetwork_t *net, float *input) {
  float *output = forward_pass(net, input);
  Layer_t *output_layer = &net->layers[net->num_layers - 1];

  int max_index = 0;
  for (int i = 1; i < output_layer->output_size; i++) {
    if (output[i] > output[max_index]) {
      max_index = i;
    }
  }

  return max_index;
}

// ===================================
//    for parallelized training loop
// ===================================

// Computes gradients without applying updates
void compute_gradients(NeuralNetwork_t *net, float *input, float *target,
                       float **gradients, float **bias_gradients) {
  // Forward pass (same as original)
  float *output = forward_pass(net, input);
  Layer_t *output_layer = &net->layers[net->num_layers - 1];

  // Allocate deltas
  float **deltas = malloc(net->num_layers * sizeof(float *));
  for (int i = 0; i < net->num_layers; i++) {
    deltas[i] = calloc(net->layers[i].output_size, sizeof(float));
  }

  // Output layer gradient (same as original)
  if (output_layer->activation == softmax_placeholder) {
    for (int i = 0; i < output_layer->output_size; i++) {
      deltas[net->num_layers - 1][i] = output_layer->output[i] - target[i];
    }
  } else {
    for (int i = 0; i < output_layer->output_size; i++) {
      float error = output_layer->output[i] - target[i];
      deltas[net->num_layers - 1][i] =
          error * output_layer->activation_derivative(output_layer->input[i]);
    }
  }

  // Backpropagation (same as original)
  for (int l = net->num_layers - 2; l >= 0; l--) {
    Layer_t *current = &net->layers[l];
    Layer_t *next = &net->layers[l + 1];

    for (int i = 0; i < current->output_size; i++) {
      float error = 0;
      for (int j = 0; j < next->output_size; j++) {
        error += next->weights[i * next->output_size + j] * deltas[l + 1][j];
      }
      deltas[l][i] = error * current->activation_derivative(current->input[i]);
    }
  }

  // Accumulate gradients instead of applying updates
  float *prev_output = input;
  for (int l = 0; l < net->num_layers; l++) {
    Layer_t *layer = &net->layers[l];

    // Accumulate weight gradients
    for (int i = 0; i < layer->input_size; i++) {
      for (int j = 0; j < layer->output_size; j++) {
        int idx = i * layer->output_size + j;
        gradients[l][idx] += prev_output[i] * deltas[l][j];
      }
    }

    // Accumulate bias gradients
    for (int j = 0; j < layer->output_size; j++) {
      bias_gradients[l][j] += deltas[l][j];
    }

    prev_output = layer->output;
  }

  // Clean up
  for (int i = 0; i < net->num_layers; i++) {
    free(deltas[i]);
  }
  free(deltas);
}

// Applies accumulated gradients
void apply_updates(NeuralNetwork_t *net, float **gradients,
                   float **bias_gradients, int batch_size) {
  for (int l = 0; l < net->num_layers; l++) {
    Layer_t *layer = &net->layers[l];
    int weights_size = layer->input_size * layer->output_size;

    // Update weights with momentum
    for (int i = 0; i < weights_size; i++) {
      float avg_gradient = gradients[l][i] / batch_size;
      layer->weight_momentum[i] = net->momentum * layer->weight_momentum[i] +
                                  net->learning_rate * avg_gradient;
      layer->weights[i] -= layer->weight_momentum[i];
    }

    // Update biases with momentum
    for (int j = 0; j < layer->output_size; j++) {
      float avg_gradient = bias_gradients[l][j] / batch_size;
      layer->bias_momentum[j] = net->momentum * layer->bias_momentum[j] +
                                net->learning_rate * avg_gradient;
      layer->biases[j] -= layer->bias_momentum[j];
    }
  }
}

/*
  Caclulates loss of a layer depending on its activation function
  Cross-entropy for softmax
  MSE for ReLU, or anything else
*/
float calculate_loss(NeuralNetwork_t *net, float *output, float *target) {
  Layer_t *output_layer = &net->layers[net->num_layers - 1];
  float loss = 0;

  // cross-entropy loss
  if (output_layer->activation == softmax_placeholder) {
    for (int i = 0; i < output_layer->output_size; i++) {
      if (target[i] > 0) {
        loss += -target[i] * logf(output[i] + 1e-10f);
      }
    }
  } else {
    // MSE loss
    for (int i = 0; i < output_layer->output_size; i++) {
      float diff = output[i] - target[i];
      loss += diff * diff;
    }
    loss /= output_layer->output_size;
  }

  return loss;
}

float softmax_placeholder(float x) { return x; }

int save_Network(NeuralNetwork_t *network, const char *filename) {
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    perror("Failed to open file for writing");
    return -1;
  }

  // Header
  uint8_t magic = MAGIC_NUMBER;
  uint32_t version = FILE_VERSION;
  fwrite(&magic, sizeof(uint8_t), 1, fp);
  fwrite(&version, sizeof(uint32_t), 1, fp);

  // Network configuration
  fwrite(&network->learning_rate, sizeof(float), 1, fp);
  fwrite(&network->momentum, sizeof(float), 1, fp);
  fwrite(&network->version, sizeof(int), 1, fp);

  // Layer count (including input layer)
  uint8_t num_layers = network->num_layers + 1;
  fwrite(&num_layers, sizeof(uint8_t), 1, fp);

  // Layer sizes (input layer first)
  uint32_t input_size = network->layers[0].input_size;
  fwrite(&input_size, sizeof(uint32_t), 1, fp);

  for (int i = 0; i < network->num_layers; i++) {
    uint32_t size = network->layers[i].output_size;
    fwrite(&size, sizeof(uint32_t), 1, fp);
  }

  // Activation functions
  for (int i = 0; i < network->num_layers; i++) {
    ActivationType act_type = ACT_RELU; // default

    if (network->layers[i].activation == relu) {
      act_type = ACT_RELU;
   } else if (network->layers[i].activation == softmax_placeholder) {
      act_type = ACT_SOFTMAX;
    }

    fwrite(&act_type, sizeof(ActivationType), 1, fp);
  }

  // Weights and biases
  for (int l = 0; l < network->num_layers; l++) {
    Layer_t *layer = &network->layers[l];
    size_t weights_size = layer->input_size * layer->output_size;
    size_t biases_size = layer->output_size;

    if (fwrite(layer->weights, sizeof(float), weights_size, fp) !=
            weights_size ||
        fwrite(layer->biases, sizeof(float), biases_size, fp) != biases_size) {
      fclose(fp);
      return -1;
    }
  }

  fclose(fp);
  return 0;
}

NeuralNetwork_t *load_Network(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    perror("Failed to open file for reading");
    return NULL;
  }

  // Read and verify header
  uint8_t magic;
  uint32_t file_version;
  if (fread(&magic, sizeof(uint8_t), 1, fp) != 1 ||
      fread(&file_version, sizeof(uint32_t), 1, fp) != 1) {
    fprintf(stderr, "Failed to read file header\n");
    fclose(fp);
    return NULL;
  }

  if (magic != MAGIC_NUMBER) {
    fprintf(stderr, "Invalid file format (bad magic number)\n");
    fclose(fp);
    return NULL;
  }

  if (file_version > FILE_VERSION) {
    fprintf(stderr, "Unsupported file version: %d (max supported: %d)\n",
            file_version, FILE_VERSION);
    fclose(fp);
    return NULL;
  }

  // Read network configuration
  float learning_rate, momentum;
  int version;
  if (fread(&learning_rate, sizeof(float), 1, fp) != 1 ||
      fread(&momentum, sizeof(float), 1, fp) != 1 ||
      fread(&version, sizeof(int), 1, fp) != 1) {
    fprintf(stderr, "Failed to read network configuration\n");
    fclose(fp);
    return NULL;
  }

  // Read layer count
  uint8_t num_layers;
  if (fread(&num_layers, sizeof(uint8_t), 1, fp) != 1) {
    fprintf(stderr, "Failed to read layer count\n");
    fclose(fp);
    return NULL;
  }

  if (num_layers < 2) {
    fprintf(stderr, "Invalid network: must have at least 2 layers\n");
    fclose(fp);
    return NULL;
  }

  // Read layer sizes
  uint32_t *layer_sizes = malloc(num_layers * sizeof(uint32_t));
  if (!layer_sizes) {
    perror("Memory allocation failed for layer sizes");
    fclose(fp);
    return NULL;
  }

  for (int i = 0; i < num_layers; i++) {
    if (fread(&layer_sizes[i], sizeof(uint32_t), 1, fp) != 1) {
      fprintf(stderr, "Failed to read layer size %d\n", i);
      free(layer_sizes);
      fclose(fp);
      return NULL;
    }
  }

  // Read activation functions
  ActivationType *act_types = malloc((num_layers - 1) * sizeof(ActivationType));
  if (!act_types) {
    perror("Memory allocation failed for activation types");
    free(layer_sizes);
    fclose(fp);
    return NULL;
  }

  for (int i = 0; i < num_layers - 1; i++) {
    if (fread(&act_types[i], sizeof(ActivationType), 1, fp) != 1) {
      fprintf(stderr, "Failed to read activation type for layer %d\n", i);
      free(layer_sizes);
      free(act_types);
      fclose(fp);
      return NULL;
    }
  }

  // Create activation function arrays
  ActivationFunc *activations =
      malloc((num_layers - 1) * sizeof(ActivationFunc));
  ActivationDerivative *derivatives =
      malloc((num_layers - 1) * sizeof(ActivationDerivative));

  if (!activations || !derivatives) {
    perror("Memory allocation failed for activation functions");
    free(layer_sizes);
    free(act_types);
    free(activations);
    free(derivatives);
    fclose(fp);
    return NULL;
  }

  for (int i = 0; i < num_layers - 1; i++) {
    switch (act_types[i]) {
    case ACT_RELU:
      activations[i] = relu;
      derivatives[i] = relu_derivative;
      break;
    case ACT_SOFTMAX:
      activations[i] = softmax_placeholder;
      derivatives[i] = NULL; // Handled specially
      break;
    default:
      fprintf(stderr, "Unknown activation type %d\n", act_types[i]);
      free(layer_sizes);
      free(act_types);
      free(activations);
      free(derivatives);
      fclose(fp);
      return NULL;
    }
  }

  // Create network
  NeuralNetwork_t *net =
      create_network((int *)layer_sizes, num_layers, activations, derivatives,
                     learning_rate, momentum);

  free(layer_sizes);
  free(act_types);
  free(activations);
  free(derivatives);

  if (!net) {
    fclose(fp);
    return NULL;
  }
  net->version = version;

  // Read weights and biases
  for (int l = 0; l < net->num_layers; l++) {
    Layer_t *layer = &net->layers[l];
    size_t weights_size = layer->input_size * layer->output_size;
    size_t biases_size = layer->output_size;

    if (fread(layer->weights, sizeof(float), weights_size, fp) !=
            weights_size ||
        fread(layer->biases, sizeof(float), biases_size, fp) != biases_size) {
      fprintf(stderr, "Failed to read layer %d parameters\n", l);
      free_network(net);
      fclose(fp);
      return NULL;
    }
  }

  fclose(fp);
  return net;
}
