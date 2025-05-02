// good practice to use const for read only variables

/*
 * a.w + b -> Relu = Activation
 */
__kernel void forward_With_Relu(__global const float *layer_Weight,
                                __global const float *layer_Bias,
                                __global float *layer_activation,
                                __global const float *layer_Input,
                                const int input_dim, const int output_dim) {
  const int i = get_global_id(0);

  float sum = layer_Bias[i];
  for (int j = 0; j < input_dim; j++) {
    sum += layer_Input[j] * layer_Weight[j * output_dim + i];
  }

  layer_activation[i] = sum >= 0.0f ? sum : 0.0f;
}

/*
 * a.w + b -> softmax = Activation
 *
 * TODO improve the softmax part, currently placed here just to prevent
 * unnecessary copy backs
 */
__kernel void forward_With_Softmax(__global const float *layer_Weight,
                                   __global const float *layer_Bias,
                                   __global float *layer_activation,
                                   __global const float *layer_Input,
                                   const int input_dim, const int output_dim) {
  const int i = get_global_id(0);

  // Compute weighted sum
  float sum = layer_Bias[i];
  for (int j = 0; j < input_dim; j++) {
    sum += layer_Input[j] * layer_Weight[j * output_dim + i];
  }
  layer_activation[i] = sum;

  // Wait for all threads to finish computing sums
  barrier(CLK_GLOBAL_MEM_FENCE);

  // Only thread 0 computes the softmax denominator
  if (i == 0) {
    float max_val = layer_activation[0];
    for (int j = 1; j < output_dim; j++) {
      if (layer_activation[j] > max_val) {
        max_val = layer_activation[j];
      }
    }

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
 * calculates gradient, activation - one hot target
 * updates bias
 * updates weight
 */
__kernel void backward_With_Softmax(
    __global float *layer_Weight, __global float *layer_Bias,
    __global float *weight_Momentum, __global float *bias_Momentum,
    __global const float *layer_Activation, __global const float *layer_Input,
    __global const float *target_Output, const int input_Size,
    const int output_Size, const float learning_Rate, const float momentum) {
  const int i = get_global_id(0);

  if (i < output_Size) {
    // calc gradient using 1 hot label
    float grad = layer_Activation[i] - target_Output[i];

    // printf("output grad, %f\n", grad);

    // Update bias
    bias_Momentum[i] = momentum * bias_Momentum[i] + learning_Rate * grad;
    layer_Bias[i] -= bias_Momentum[i];

    // Update weights
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
  calculates gradient, ReLU derivation
  updates bias
  updates weight
 */
__kernel void backward_With_Relu(
    __global float *layer_Weight, __global float *layer_Bias,
    __global float *weight_Momentum, __global float *bias_Momentum,
    __global const float *next_Gradient, __global const float *layer_Activation,
    __global const float *layer_Input, const int input_Size,
    const int output_Size, const float learning_Rate, const float momentum) {

  const int i = get_global_id(0);

  // printf("Next_Gradient[%d], %f\n", i, next_Gradient[i]);

  if (i < output_Size) {
    // calc gradient using relu derivative
    float grad = (layer_Activation[i] > 0.0f) ? next_Gradient[i] : 0.0f;

    // Update bias
    bias_Momentum[i] = momentum * bias_Momentum[i] + learning_Rate * grad;
    layer_Bias[i] -= bias_Momentum[i];

    // Update weights
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
  Calculates the gradient to be passed to the previous layer.
  This is d(loss) / d(input of current layer), which is equivalent to
  d(loss) / d(output of previous layer).

  previous_Gradient[i] = sum(current_Weights[i * current_OutputSize + j] *
  next_Gradient[j]) for all j
 */
__kernel void weight_gradient(
    __global const float *current_Weights, __global const float *next_Gradient,
    __global float *previous_Gradient,
    const int current_InputSize, // Output size of the previous layer
    const int current_OutputSize // Output size of the current layer
) {
  const int i = get_global_id(0); // Neuron index in the previous layer

  if (i < current_InputSize) {
    float grad_sum = 0.0f;
    for (int j = 0; j < current_OutputSize;
         j++) { // Iterate over neurons in the current layer
      grad_sum +=
          current_Weights[i * current_OutputSize + j] * next_Gradient[j];
    }
    previous_Gradient[i] = grad_sum;
  }
}

__kernel void compute_output_gradient(__global const float *activation,
                                      __global float *gradient,
                                      __global const float *target,
                                      const int size) {
  int i = get_global_id(0);
  if (i < size) {
    gradient[i] = activation[i] - target[i];
  }
}
