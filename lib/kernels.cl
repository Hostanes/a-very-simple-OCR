
// good practice to use const for read only variables

/*
  a.w + b -> Relu = Activation
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
  a.w + b -> softmax = Activation

  TODO improve the softmax part, currently placed here just to prevent
  unnecessary copy backs
*/
__kernel void forward_With_Softmax(__global const float *layer_Weight,
                                   __global const float *layer_Bias,
                                   __global float *layer_activation,
                                   __global const float *layer_Input,
                                   const int input_dim, const int output_dim) {
  const int i = get_global_id(0);

  // matmult section

  float sum = layer_Bias[i];
  for (int j = 0; j < input_dim; j++) {
    sum += layer_Input[j] * layer_Weight[j * output_dim + i];
  }

  // Softmax section

  float max_val = -FLT_MAX;
  for (int j = 0; j < output_dim; j++) {
    max_val = max(max_val, sum);
  }

  float exp_val = exp(sum - max_val);

  float exp_sum = 0.0f;
  for (int j = 0; j < output_dim; j++) {
    float temp_sum = layer_Bias[j];
    for (int k = 0; k < input_dim; k++) {
      temp_sum += layer_Input[k] * layer_Weight[k * output_dim + j];
    }
    exp_sum += exp(temp_sum - max_val);
  }

  layer_activation[i] = exp_val / exp_sum;
}

/*
  calculates gradient, activation - one hot target
  updates bias
  updates weight
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
