/*
  nn-kernel.cl

*/

// Forward pass kernel
__kernel void
forward_layer(__global const float *input, __global const float *weights,
              __global const float *biases, __global float *output,
              const int input_size, const int output_size,
              const int activation_type // 0: ReLU, 1: Linear (for softmax)
) {
  int j = get_global_id(0);
  if (j >= output_size)
    return;

  float sum = biases[j];
  for (int k = 0; k < input_size; k++) {
    sum += input[k] * weights[k * output_size + j];
  }

  if (activation_type == 0) {
    output[j] = fmax(0.0f, sum); // ReLU
  } else {
    output[j] = sum; // Linear (for softmax)
  }
}

// Backward pass kernel for output layer (softmax with cross-entropy)
__kernel void backward_output_layer(__global const float *output,
                                    __global const float *target,
                                    __global float *deltas,
                                    const int output_size) {
  int j = get_global_id(0);
  if (j >= output_size)
    return;

  // For softmax with cross-entropy, gradient is simply output - target
  deltas[j] = output[j] - target[j];
}

// Backward pass kernel for hidden layers (ReLU)
__kernel void backward_hidden_layer(__global const float *next_weights,
                                    __global const float *next_deltas,
                                    __global const float *input,
                                    __global float *deltas,
                                    const int current_size,
                                    const int next_size) {
  int j = get_global_id(0);
  if (j >= current_size)
    return;

  float error = 0.0f;
  for (int k = 0; k < next_size; k++) {
    error += next_weights[j * next_size + k] * next_deltas[k];
  }

  // ReLU derivative: 1 if input > 0, else 0
  deltas[j] = error * (input[j] > 0.0f ? 1.0f : 0.0f);
}

// Weight update kernel
__kernel void
update_weights(__global const float *input, __global const float *deltas,
               __global float *weights, __global float *weight_momentum,
               __global float *biases, __global float *bias_momentum,
               const float learning_rate, const float momentum,
               const int input_size, const int output_size) {
  int i = get_global_id(0);
  int j = get_global_id(1);

  if (i >= input_size || j >= output_size)
    return;

  int idx = i * output_size + j;
  float gradient = input[i] * deltas[j];

  // Update weight momentum and weight
  weight_momentum[idx] =
      momentum * weight_momentum[idx] + learning_rate * gradient;
  weights[idx] -= weight_momentum[idx];

  // Update bias (only for the first row of threads)
  if (i == 0) {
    bias_momentum[j] = momentum * bias_momentum[j] + learning_rate * deltas[j];
    biases[j] -= bias_momentum[j];
  }
}
