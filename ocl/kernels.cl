__kernel void forward_pass(
    const int input_size,
    const int output_size,
    __global const float* weights,
    __global const float* biases,
    __global const float* input,
    __global float* output
) {
    const int i = get_global_id(0);
    
    if (i < output_size) {
        float sum = biases[i];
        
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weights[j * output_size + i];
        }
        
        // ReLU activation
        output[i] = sum > 0.0f ? sum : 0.0f;
    }
}

__kernel void softmax(
    const int size,
    __global float* data
) {
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        data[i] = exp(data[i] - max_val);
        sum += data[i];
    }
    
    for (int i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

__kernel void backward_output(
    const int hidden_size,
    const int output_size,
    __global const float* hidden_output,
    __global const float* output_grad,
    __global float* weights,
    __global float* weight_momentum,
    __global float* bias_momentum,
    const float learning_rate,
    const float momentum
) {
    const int i = get_global_id(0);
    
    if (i < output_size) {
        bias_momentum[i] = momentum * bias_momentum[i] + learning_rate * output_grad[i];
        
        for (int j = 0; j < hidden_size; j++) {
            int idx = j * output_size + i;
            float grad = output_grad[i] * hidden_output[j];
            weight_momentum[idx] = momentum * weight_momentum[idx] + learning_rate * grad;
            weights[idx] -= weight_momentum[idx];
        }
    }
}

__kernel void hidden_gradients(
    const int hidden_size,
    const int output_size,
    __global const float* hidden_output,
    __global const float* output_grad,
    __global const float* weights,
    __global float* hidden_grad
) {
    const int j = get_global_id(0);
    
    if (j < hidden_size) {
        float sum = 0.0f;
        
        for (int i = 0; i < output_size; i++) {
            sum += output_grad[i] * weights[j * output_size + i];
        }
        
        // Apply ReLU derivative
        hidden_grad[j] = (hidden_output[j] > 0.0f) ? sum : 0.0f;
    }
}

__kernel void backward_hidden(
    const int input_size,
    const int hidden_size,
    __global const float* input,
    __global const float* hidden_grad,
    __global float* weights,
    __global float* weight_momentum,
    __global float* bias_momentum,
    const float learning_rate,
    const float momentum
) {
    const int i = get_global_id(0);
    
    if (i < hidden_size) {
        bias_momentum[i] = momentum * bias_momentum[i] + learning_rate * hidden_grad[i];
        
        for (int j = 0; j < input_size; j++) {
            int idx = j * hidden_size + i;
            float grad = hidden_grad[i] * input[j];
            weight_momentum[idx] = momentum * weight_momentum[idx] + learning_rate * grad;
            weights[idx] -= weight_momentum[idx];
        }
    }
}
