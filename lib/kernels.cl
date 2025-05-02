// Improved OpenCL kernels for neural network operations

/*
 * Forward pass with ReLU activation
 * a.w + b -> ReLU = Activation
 */
__kernel void forward_With_Relu(__global const float *layer_Weight,
                                __global const float *layer_Bias,
                                __global float *layer_activation,
                                __global const float *layer_Input,
                                const int input_dim, const int output_dim) {
  const int i = get_global_id(0);

  if (i < output_dim) {
    float sum = layer_Bias[i];
    for (int j = 0; j < input_dim; j++) {
      sum += layer_Input[j] * layer_Weight[j * output_dim + i];
    }

    // ReLU activation function
    layer_activation[i] = sum > 0.0f ? sum : 0.0f;
  }
}

/*
 * Forward pass with Softmax activation
 * a.w + b -> softmax = Activation
 */
__kernel void forward_With_Softmax(__global const float *layer_Weight,
                                   __global const float *layer_Bias,
                                   __global float *layer_activation,
                                   __global const float *layer_Input,
                                   const int input_dim, const int output_dim) {
  const int i = get_global_id(0);

  if (i < output_dim) {
    // Compute weighted sum
    float sum = layer_Bias[i];
    for (int j = 0; j < input_dim; j++) {
      sum += layer_Input[j] * layer_Weight[j * output_dim + i];
    }
    layer_activation[i] = sum;
  }

  // Wait for all threads to finish computing sums
  barrier(CLK_GLOBAL_MEM_FENCE);

  // Only thread 0 computes the softmax
  if (i == 0) {
    // Find max value for numerical stability
    float max_val = layer_activation[0];
    for (int j = 1; j < output_dim; j++) {
      if (layer_activation[j] > max_val) {
        max_val = layer_activation[j];
      }
    }

    // Compute exp(x - max) and sum
    float exp_sum = 0.0f;
    for (int j = 0; j < output_dim; j++) {
      float exp_val = exp(layer_activation[j] - max_val);
      layer_activation[j] = exp_val;
      exp_sum += exp_val;
    }

    // Normalize
    for (int j = 0; j < output_dim; j++) {
      layer_activation[j] /= exp_sum;
    }
  }
}

/*
 * Compute output gradient: activation - target
 */
__kernel void compute_output_gradient(__global const float *activation,
                                      __global float *gradient,
                                      __global const float *target,
                                      const int size) {
  int i = get_global_id(0);

  if (i < size) {
    // For softmax with cross-entropy loss, gradient is simply (activation -
    // target)
    gradient[i] = activation[i] - target[i];
  }
}

/*
 * Backward pass for output layer with Softmax activation
 * Updates weights and biases based on gradients
 */
__kernel void backward_With_Softmax(
    __global float *layer_Weight, __global float *layer_Bias,
    __global float *weight_Momentum, __global float *bias_Momentum,
    __global const float
        *layer_Gradient, // Using explicit gradient instead of computing it here
    __global const float *layer_Input, __global const float *target_Output,
    const int input_Size, const int output_Size, const float learning_Rate,
    const float momentum) {

  const int i = get_global_id(0);

  if (i < output_Size) {
    // Use the pre-computed gradient
    float grad = layer_Gradient[i];

    // Update bias with momentum
    bias_Momentum[i] = momentum * bias_Momentum[i] + learning_Rate * grad;
    layer_Bias[i] -= bias_Momentum[i];

    // Update weights with momentum
    for (int j = 0; j < input_Size; j++) {
      int idx = j * output_Size + i;
      float w_grad = grad * layer_Input[j];
      weight_Momentum[idx] =
          momentum * weight_Momentum[idx] + learning_Rate * w_grad;
      layer_Weight[idx] -= weight_Momentum[idx];
    }
  }
}

/*
 * Calculate gradient to be propagated to previous layer
 * Computes d(loss)/d(input) which is the gradient for the previous layer
 */
__kernel void
weight_gradient(__global const float *current_Weights,
                __global const float *next_Gradient,
                __global float *previous_Gradient,
                const int current_InputSize, // Output size of prev layer
                const int current_OutputSize // Output size of current layer
) {
  const int i = get_global_id(0); // Neuron index in previous layer

  if (i < current_InputSize) {
    float grad_sum = 0.0f;
    for (int j = 0; j < current_OutputSize; j++) {
      // Fixed indexing: weights are stored as [input_neuron][output_neuron]
      grad_sum +=
          current_Weights[i * current_OutputSize + j] * next_Gradient[j];
    }
    previous_Gradient[i] = grad_sum;
  }
}

/*
 * Backward pass for hidden layer with ReLU activation
 * Updates weights and biases based on gradients
 */
__kernel void backward_With_Relu(
    __global float *layer_Weight, __global float *layer_Bias,
    __global float *weight_Momentum, __global float *bias_Momentum,
    __global const float
        *layer_Gradient, // Already calculated gradient from weight_gradient
    __global const float *layer_Activation, __global const float *layer_Input,
    const int input_Size, const int output_Size, const float learning_Rate,
    const float momentum) {

  const int i = get_global_id(0);

  if (i < output_Size) {
    // Apply ReLU derivative: if activation > 0 then 1, else 0
    float relu_deriv = (layer_Activation[i] > 0.0f) ? 1.0f : 0.0f;

    // Multiply incoming gradient by ReLU derivative
    float grad = layer_Gradient[i] * relu_deriv;

    // Update bias with momentum
    bias_Momentum[i] = momentum * bias_Momentum[i] + learning_Rate * grad;
    layer_Bias[i] -= bias_Momentum[i];

    // Update weights with momentum
    for (int j = 0; j < input_Size; j++) {
      int idx = j * output_Size + i;
      float w_grad = grad * layer_Input[j];
      weight_Momentum[idx] =
          momentum * weight_Momentum[idx] + learning_Rate * w_grad;
      layer_Weight[idx] -= weight_Momentum[idx];
    }
  }
}
