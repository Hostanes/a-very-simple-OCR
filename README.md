
## Parallelized Neural Network - Phase 3 Report

---

## Check list

- [x] Serial Code
- [x] OpenMP parallelism over propagation
- [x] OpenCL parallelism over propagation (TODO has to be rewritten)
- [ ] MPI parallelism over propagation
## How to use

I have included a selection of bash scripts to simplify the compilation and running process, those being:

- `./compile-and-run.sh serial/mnist-test.c` runs the train and evaluate on the mnist dataset using single threading
- `./compile-and-run-omp.sh omp/mnist-test.c` runs the train and evaluate on the mnist dataset using omp parallelism with max threads.
- `./compile-and-run-ocl.sh ocl/mnist-test-ocl.c` runs the train and evaluate on the mnist dataset using ocl GPU parallelism
- `./train-and-compare.sh` runs all the above 3, each of them produces a `*.nn` file saving the network info and weights. After training all of them it tests all 3 networks on a single image, displaying the confidence of each in their output.


***Note:***
	This Dense Neural Network library was built with the intention of being modular and dynamic. However handling the complicated buffer allocations and management inside dynamic structs proved to difficult and cumbersome to manage. Therefore the current OpenCL implementation is 1 long C script with the kernels stored in a separate kernels.c. It does not allow custom network architectures, it is not scalable. And it is barely readable. But it does function.
	We intend to rewrite this in a way that suits the initial intent behind this project. That being a simple scalable parallelized DNN library. Using more efficient device memory management, parallel swapping in and out of weight buffers, and inplace activation buffers (ping pong method).

## Parallelism in OpenCL case

The main concern in OpenCL parallelism is to minimize data transfers between Device memory and CPU memory. To accomplish this all necessary weight, bias, and output buffers are stored in the device at the beginning. 

therefore we store all of this information in global variables at the top of a single fine.

```
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel forward_kernel, softmax_kernel, backward_output_kernel,
    hidden_gradients_kernel, backward_hidden_kernel;
cl_mem d_hidden1_weights, d_hidden1_biases, d_hidden1_weight_momentum,
    d_hidden1_bias_momentum;
cl_mem d_hidden2_weights, d_hidden2_biases, d_hidden2_weight_momentum,
    d_hidden2_bias_momentum;
cl_mem d_output_weights, d_output_biases, d_output_weight_momentum,
    d_output_bias_momentum;
cl_mem d_input, d_hidden1_output, d_hidden2_output, d_output, d_output_grad,
    d_hidden2_grad, d_hidden1_grad;

typedef struct {
  float *weights, *biases, *weight_momentum, *bias_momentum;
  int input_size, output_size;
} Layer;

typedef struct {
  Layer hidden1, hidden2, output;
} Network;

typedef struct {
  unsigned char *images, *labels;
  int nImages;
} InputData;

```

The parallel kernels are stored in the `kernels.cl` file. They are

- `__kernel void forward_pass`: runs the $a . w + b$ into ReLU
- `__kernel void softmax`: runs Softmax on an input float buffer
- `__kernel void backward_output`: Updates the weights for the final output Softmax layers
- `__kernel void hidden_gradients`: Calculates the gradients for the internal hidden ReLU layers
- `__kernel void backward_hidden`: Updates the weights for the internal hidden ReLU layers

The forward matrix multiplication does not use tiling, as it did not have a measurable improvement over naive matrix multiplication in our case of matrix by vector dot product with relatively small dimensions.
However to test the effectiveness of this we included a `tiling-vs-naive-ocl.c` script and an omp version in the `tests` folder. Where in both cases the tiling code showcased much faster runtime and a significantly lower **L1 cache miss rate**.

These kernels are all called on the previous stored cl_mem buffers without any unnecessary **copys** and **copy-backs**

However, our code still displays a severe slowdown compared to both the OpenMP code and the serial code. This could be due to our hardware setup, utilizing the integrated GPU of a ryzen 7 5850u cpu instead of a dedicated GPU. We need to test this on a more powerful system with a dGPU to confirm.

## Parallelism in OpenMP case


Simply parallelize the operations inside the each forward backward propagation cycle, these are:

Forward dot product:
```
#pragma omp parallel for
    for (int j = 0; j < layer->output_size; j++) {
      float sum = layer->biases[j];
      for (int k = 0; k < layer->input_size; k++) {
        sum += current_input[k] * layer->weights[k * layer->output_size + j];
      }
      layer->input[j] = sum;
    }
```
Note that with large matrix multiplication it would be better to use tiling. As it decreases expensive L1 cache misses. However in our MNIST implementation with small $Matrix \times Vector$ operations the added overhead only increases runtime. 
For a comparison of both methods see the standalone C script `tiling-vs-naive.c`

Forward activation function:
```
#pragma omp parallel for
      for (int j = 0; j < layer->output_size; j++) {
        layer->output[j] = layer->activation(layer->input[j]);
      }
```

Output layer error calculation:
```
#pragma omp parallel for
for (int i = 0; i < output_layer->output_size; i++) {
    deltas[net->num_layers - 1][i] = output_layer->output[i] - target[i];
}
```

Hidden layer error propagation:
```
#pragma omp parallel for
for (int i = 0; i < current->output_size; i++) {
    float error = 0;
    for (int j = 0; j < next->output_size; j++) {
        error += next->weights[i * next->output_size + j] * deltas[l + 1][j];
    }
    deltas[l][i] = error * current->activation_derivative(current->input[i]);
}
```

Weight and bias updates:
```
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

#pragma omp parallel for
for (int j = 0; j < layer->output_size; j++) {
    layer->bias_momentum[j] = net->momentum * layer->bias_momentum[j] +
                            net->learning_rate * deltas[l][j];
    layer->biases[j] -= layer->bias_momentum[j];
}
```

All of these operations have the advantage of being trivially parallel with no chance of overwrites or race conditions, negating the need for locks, reduction pragma, or atomic pragma operations which could cause slow down or complications


## Sample runs

```
Compiling mnist-train.c with lib/nnlib.c...
Running mnist-test-serial to generate serial.nn...

Random Test Image (True Label: 5):

                .@#
                @@@.
             ..#@@.
     *@***#@@@@@*
     *@@@@@@@#..
      #@@**.
       #@*....
        @@@@@@@@@*
        #@@#####@@@.
                 .@@.
                  *@@
                   *@.
                    @@
                    #@*
                    .@*
         .*         .@*
         *#         #@*
         #@.     .*@@@
          *@@@*@@@@@#
            ###@@*..



Starting training...
Model will be saved to: serial.nn
Epoch 1, Accuracy: 95.62%, Avg Loss: 0.2726, Time: 69.43 seconds
Epoch 2, Accuracy: 96.70%, Avg Loss: 0.1027, Time: 69.34 seconds
Epoch 3, Accuracy: 97.08%, Avg Loss: 0.0626, Time: 69.07 seconds
Epoch 4, Accuracy: 97.63%, Avg Loss: 0.0394, Time: 68.98 seconds
Epoch 5, Accuracy: 97.73%, Avg Loss: 0.0263, Time: 69.16 seconds
Predicted Label: 5
Model successfully saved to serial.nn
serial.nn generated successfully

Compiling mnist-train.c with lib/nnlib-omp.c...
Running mnist-test-parallel to generate omp.nn...

Random Test Image (True Label: 9):


           *#@@.
         #@@@@@#
        #@@*  #@
       #@#    *@
       @@.   .@@@
       @@   .@@@@
       @@#*@@@@@@.
       *@@@@#* @@#
          .    @@#
               *@@
               *@@
                @@
                @@.
                @@.
                @@.
                @@*
                *@#
                *@@
                *@@
                 @#


Starting training...
Model will be saved to: omp.nn
Epoch 1, Accuracy: 96.40%, Avg Loss: 0.2746, Time: 16.88 seconds
Epoch 2, Accuracy: 97.33%, Avg Loss: 0.1038, Time: 16.94 seconds
Epoch 3, Accuracy: 97.63%, Avg Loss: 0.0652, Time: 19.06 seconds
Epoch 4, Accuracy: 97.69%, Avg Loss: 0.0420, Time: 17.50 seconds
Epoch 5, Accuracy: 97.73%, Avg Loss: 0.0270, Time: 21.04 seconds
Predicted Label: 9
Model successfully saved to omp.nn
omp.nn generated successfully

Random Test Image (True Label: 5):

           #*
          #@@#*@@@####.
         #@@@@@@@@@@@@@
         @@@@@@#*@@#**#.
        *@**.
        #*
        ## ..
        #@@@@**
         @@#**@@@#
                #@*
                  #
                   #
                   #
                   #
                  ..
         *       .@
         *@*.  .#@#
          .@@@@@@@.
            #@@@@*
             .@*


Dataset: 48000 training samples, 12000 test samples
Epoch 1, Accuracy: 95.41%, Avg Loss: 0.2861, Time: 110.41 seconds
Epoch 2, Accuracy: 96.38%, Avg Loss: 0.1103, Time: 110.14 seconds
Epoch 3, Accuracy: 96.88%, Avg Loss: 0.0690, Time: 110.38 seconds
Epoch 4, Accuracy: 96.88%, Avg Loss: 0.0457, Time: 110.01 seconds
Predicted Label: 5
Total time needed: 440.94 seconds
Model successfully saved to ocl.nn


Selected Test Image (True Label: 0):
          .#@@@@@@..
         #@@@@@@@@@@#
        #@@@#****#@@@@*
        @@#.      *@@@@#
       #@@         *#@@@.
      #@@@           #@@.
      #@@*           .@@.
     .@@@            .@@.
     .@@*            #@@.
     *@@.           #@@#
     @@@.          #@@@#
     @@@.         .@@@#
     @@@.         @@@@
     .@@*       #@@@@*
     .@@@      #@@@@*
      #@@#    #@@@@*
       #@@@..#@@@#
        *@@#@@@@#
          @@@@@#
         .@@@@.

Model Output Comparison:
Class      Serial          OpenMP          OpenCL
------------------------------------------------------------
0          1.000000        1.000000        1.000000
1          0.000000        0.000000        0.000000
2          0.000000        0.000000        0.000000
3          0.000000        0.000000        0.000000
4          0.000000        0.000000        0.000000
5          0.000000        0.000000        0.000000
6          0.000000        0.000000        0.000000
7          0.000000        0.000000        0.000000
8          0.000000        0.000000        0.000000
9          0.000000        0.000000        0.000000

Predictions:
Serial Model:  0 (confidence: 100.00%)
OpenMP Model:  0 (confidence: 100.00%)
OpenCL Model:  0 (confidence: 100.00%)
True Label:    0

```
