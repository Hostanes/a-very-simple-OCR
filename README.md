```
 A dense neural network library written in C, allows for custom architectures and activation functions.

 Tested on random data and the MNIST handwritten character dataset

 Main neural network functionality is modified from:
 https://github.com/konrad-gajdus/miniMNIST-c
```

 Parallelized using:
 - [ x ] OpenMP
 - [  ] MPI
 - [  ] OpenCL


## How to use

to train, compile, and compare both serial and parallel scripts, simply execute the `train-and-compare.sh` bash script

**list of bash scripts**
- `compile-and-run.sh` :  compiles and runs a C script using the `nnlib.c` library
- `compile-and-run-par.sh` :  compiles and runs a C script using the `nnlib-par.c` library
- `train-and-compare.sh` : runs both of the above commands on `mnist-train.c` script and compares both using the `mnist-model-tester.c`

**list of C scripts**
- `nnlib.h` : the header file, includes used structs and function declarations
- `nnlib.c` :  a serial implementation of the header file
- `nnlib-par.c` :  a parallelized  implementation of the header file
- `mnist-train.c`  :  Loads the MNIST dataset and trains it using either `nnlib.c` or `nnlib-par.c` depending on what it was compiled with, saves a `{name}.nn` file for the trained model info, name is taken from the passed arguement
- `mnist-model-tester.c`, loads 2 models, `parallel.nn` and `serial.nn` and compares them on 1 randomly selected MNIST image

**Sample output of**`train-and-compare.sh`:

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
Compiling mnist-train.c with lib/nnlib-par.c...
Running mnist-test-parallel to generate parallel.nn...

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
Model will be saved to: parallel.nn
Epoch 1, Accuracy: 96.40%, Avg Loss: 0.2746, Time: 16.88 seconds
Epoch 2, Accuracy: 97.33%, Avg Loss: 0.1038, Time: 16.94 seconds
Epoch 3, Accuracy: 97.63%, Avg Loss: 0.0652, Time: 19.06 seconds
Epoch 4, Accuracy: 97.69%, Avg Loss: 0.0420, Time: 17.50 seconds
Epoch 5, Accuracy: 97.73%, Avg Loss: 0.0270, Time: 21.04 seconds
Predicted Label: 9
Model successfully saved to parallel.nn
parallel.nn generated successfully
Compiling model tester...
Testing models...

Selected Test Image (True Label: 1):

           .@@
           .@@
           .@@
           .@@#
           .@@@.
           .@@@*
           .@@@@
            #@@@
            #@@@
            #@@@
            .@@@
             @@@#
             @@@@
             @@@@
             @@@@
             *@@@
              @@@
              @@@*
              #@@*
               #@*

Model Comparison:
Class      Parallel Output Serial Output
----------------------------------------
0          0.000000        0.000000
1          0.999972        0.999744
2          0.000001        0.000000
3          0.000002        0.000001
4          0.000014        0.000194
5          0.000000        0.000000
6          0.000000        0.000001
7          0.000009        0.000006
8          0.000003        0.000055
9          0.000000        0.000000

Predictions:
Parallel Model: 1 (confidence: 100.00%)
Serial Model:   1 (confidence: 99.97%)
True Label:     1
All tasks completed successfully
```
