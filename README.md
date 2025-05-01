```
 A dense neural network library written in C, allows for custom architectures and activation functions.

 Tested on random data and the MNIST handwritten character dataset

 Main neural network functionality is modified from:
 https://github.com/konrad-gajdus/miniMNIST-c
```

 Parallelized using:
 - [ x ] OpenMP
 - [  ] MPI
 - [ x ] OpenCL (got issues that need to fixed tho)


## How to use

I have included a selection of bash scripts to simplify the compilation and running process, those being:

- `./compile-and-run.sh mnist-test.c` runs the train and evaluate on the mnist dataset using single threading
- `./compile-and-run-omp.sh mnist-test.c` runs the train and evaluate on the mnist dataset using omp parallelism with max threads.
- `./compile-and-run-ocl.sh mnist-test-ocl.c` runs the train and evaluate on the mnist dataset using ocl GPU parallelism
- `./train-and-compare.sh` runs all the above 3, each of them produces a `*.nn` file saving the network info and weights. After training all of them it tests all 3 networks on a single image, displaying the confidence of each in their output.

## Parallelism in OpenCL case

The main concern in OpenCL parallelism is to minimize data transfers between Device memory and CPU memory. To accomplish this all necessary weight, bias, and output buffers are stored in the device at the beginning of each propagation cycle. 
Pointers to these are present in each `Layer_t` struct.

```
typedef struct {
  float *weights;
  float *biases;
  float *weight_momentum;
  float *bias_momentum;
  float *output;
  float *input;

  // Device buffers
  cl_mem weights_buf;
  cl_mem biases_buf;
  cl_mem weight_momentum_buf;
  cl_mem bias_momentum_buf;
  cl_mem output_buf;
  cl_mem input_buf;

  int input_size;
  int output_size;
  ActivationFunc activation;
  ActivationDerivative activation_derivative;
} Layer_t;
```

Then the following operations are ran on these device-side buffers, without any read backs until the end:
- Forward propagation
- Backward propagation (ReLU and Softmax layers are separate)
- Activation functions
- Activation derivatives
- Weight updates

This ensures the minimal amount of data transfer between CPU and Device buffers.

Our results below display a severe slowdown in the OpenCL test (slower than even the serial code), aswell as a measurable decrease in accuracy; we believe these are the causes:
1. **Utilizing an Integrated GPU instead of a dedicated one:**
	The tests below were ran on a laptop with only a Ryzen 7 5850u CPU and no dedicated GPU. Possibly causing the massive slowdowns witnessed below.
2. **Differences in floating point operations:**
	The lower accuracy is possibly caused by the lower precision in floating point operations used in the iGPU.

To investigate both of these issues we plan on testing this on a device with a powerful dedicated GPU.

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

Compilation successful. Running program...
Random Test Image (True Label: 3):





           .***
        .*@@@@@@
       *@@*  .@@
     *@@*    *@*
     #.     .@@
           .@@
          *@@.. ....
         #@@@@@@@@@@@@*
       *@@@@@#**...**#@@.
     .@@@#            .@@
      #*               @@
       #               @@
       @              @@*
       @#           *@@*
       *@#.       .#@@.
        *@@#*. .#@@@#
          #@@@@@@#..
            .***





Starting training...
Model will be saved to: ocl.nn
Epoch 1/5
Epoch 1, Accuracy: 76.16%, Avg Loss: -nan, Time: 184.38 seconds
Epoch 2/5
Epoch 2, Accuracy: 82.14%, Avg Loss: -nan, Time: 184.83 seconds
Epoch 3/5
Epoch 3, Accuracy: 84.56%, Avg Loss: -nan, Time: 185.05 seconds
Epoch 4/5
Epoch 4, Accuracy: 85.50%, Avg Loss: -nan, Time: 184.74 seconds
Epoch 5/5
Epoch 5, Accuracy: 86.31%, Avg Loss: -nan, Time: 185.67 seconds
Predicted Label: 3
Model successfully saved to ocl.nn

Selected Test Image (True Label: 7):


            @@@@@@#*
           @@@@@@@@@@.
          .@@@@* *@@@@*
          *@@#     @@@@
         .@@@.    *@@@*
         .@@*    *@@@#
         .@@    #@@@#
          .     @@@#
               #@@*
              *@@*
             #@@#
            *@@@
           .@@@.
          .@@@*
         .@@@*
         @@@@
        *@@@.
       .@@@*
       #@@@
       .@@


Model Output Comparison:
Class      Serial          OpenMP          OpenCL
------------------------------------------------------------
0          0.000001        0.000001        0.042977
1          0.000000        0.000000        0.018758
2          0.000001        0.000002        0.014460
3          0.000004        0.000003        0.014878
4          0.000000        0.000000        0.059687
5          0.000000        0.000000        0.060225
6          0.000000        0.000000        0.034457
7          0.995868        0.999502        0.557351
8          0.000004        0.000000        0.030665
9          0.004120        0.000491        0.166542

Predictions:
Serial Model:  7 (confidence: 99.59%)
OpenMP Model:  7 (confidence: 99.95%)
OpenCL Model:  7 (confidence: 55.74%)
True Label:    7

```
