

# Dense Neural Network: Parallelized with OpenMP

## Parallelization methods

### Introduction

In a neural network the serial pipeline goes as follows:
```
Input 1 -> Forward propagation -> Loss calculation -> Backward propagation
Input 2 -> Forward propagation -> Loss calculation -> Backward propagation
Input 3 -> Forward propagation -> Loss calculation -> Backward propagation
```

- **Forward Propagation's** main bottle neck is the large matrix dot product operation, this can be parallelized either naively, or improved with tiling. refer to section [[#Matrix dot product tiling]]
- **Backward Propagation** itself is ran serially
- **The Training loop** CAN be parallelized using **Hogwild Stochastic Gradient Descent**[^1]

---

### How to use:

All scripts include the file `config.h`
```
config.h:

#ifndef CONFIG_H
#define CONFIG_H

// Uncomment to enable OpenMP
#define USE_OMP

#endif
```

all pragmas are wrapped with:
```
#ifdef USE_OMP
#pragma omp parallel...
#endif
```

If `USE_OMP` is not defined, the C pre-processor will ignore the pragmas, effectively using single threading.

The neural network library:

- `cnnlib.c` includes the implementation of the training loop functions
- `cnnlib.h` describes the function declarations and architecture structs


Main scripts:
- `dot-prod-test.c` tests the difference between naive parallel dot product and tiled dot product
- `random-nn-test.c` tests the neural network on a randomized dataset, accuracy and loss are ignore, used to test the speed gain of parallelism
- `left-right-nn.c` runs the neural network on a generated simple dataset of 2 classes, 1 class has pixels on the left, 1 class has pixels on the right, used to sanity test weight update logic
- `mnist-nn.c` runs the neural network on the **mnist handwritten character** dataset


---

### Matrix dot product tiling:

$$A . B = C$$

Without tiling
from Matrix B, we need to use the column to calculate the dot product value at a specific index. However, the cache is loaded with the row due to spatial locality. Resulting in unnecessary loading and many expensive L1 cache misses.

![[Pasted image 20250412150105.png]]

To fix this, each thread divides the output matrix into blocks, and calculates the partial dot product for this specific block, results in the thread loading only the required data into cache 
Tiling using `BLOCK_SIZE = 2`
![[Pasted image 20250412145904.png]]

**Tiling code**


Only use the Tiling method if the matricies are large enough
```
  const int BLOCK_SIZE =
      64; // too large and blocks larger than matricies, too small and no point

  // Only parallelize for large enough matrices
  if (mat1->rows * mat2->columns > 10000) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < mat1->rows; i += BLOCK_SIZE) {
      for (int j = 0; j < mat2->columns; j += BLOCK_SIZE) {
        for (int k = 0; k < mat1->columns; k += BLOCK_SIZE) {
          // Process block
          for (int ii = i; ii < i + BLOCK_SIZE && ii < mat1->rows; ii++) {
            for (int jj = j; jj < j + BLOCK_SIZE && jj < mat2->columns; jj++) {
              double sum = result->data[ii][jj];
              for (int kk = k; kk < k + BLOCK_SIZE && kk < mat1->columns;
                   kk++) {
                sum += mat1->data[ii][kk] * mat2->data[kk][jj];
              }
              result->data[ii][jj] = sum;
            }
          }
        }
      }
    }
  } else {
    // Sequential version for small matrices
    for (int i = 0; i < mat1->rows; i++) {
      for (int j = 0; j < mat2->columns; j++) {
        double sum = 0;
        for (int k = 0; k < mat1->columns; k++) {
          sum += mat1->data[i][k] * mat2->data[k][j];
        }
        result->data[i][j] = sum;
      }
    }
  }

```

##### Testing

`dot-product-test.c` compares the runtimes of both naively parallel matrix multiplication and tiled matrix multipliciation.

using the utility `perf`, we can check the number of cache misses using this command

```
perf stat -e cache-misses ./test.o
```

where `test.o` is the compiled script

![[Pasted image 20250412152640.png]]
![[Pasted image 20250412152718.png]]

\- Fig(1)Cache misses with different parameters. Fig(2) Runtime (seconds) with different parameters. Naive matmult, tiled with different block sizes, block size is a very important parameter that depends on the matricies used.

**Naive**
```
 Performance counter stats for './dot-prod-test-naive.o':

    71,825,303,489      cache-misses

      57.913302197 seconds time elapsed

     839.460494000 seconds user
      20.974593000 seconds sys
```

**Tiled using block size 64**

```
 Performance counter stats for './dot-prod-test-tiled.o':

    59,408,664,837      cache-misses

      39.720091738 seconds time elapsed

     562.711937000 seconds user
      25.121151000 seconds sys
```

**Tiled using block size 128**

```
 Performance counter stats for './dot-prod-test-tiled.o':

    61,611,134,570      cache-misses

      40.476652730 seconds time elapsed

     577.200556000 seconds user
      26.076152000 seconds sys
```

**Tiled using block size 512**

```
 Performance counter stats for './dot-prod-test-tiled.o':

    67,027,320,265      cache-misses

      50.135536647 seconds time elapsed

     723.062333000 seconds user
      25.604815000 seconds sys
```



### Neural Network Forward Propagation

##### Basic nested for loops

In many locations there are **embarrassingly parallel** for loops or nested for loops such as:

**Copying data from one matrix to another, only done for large enough matricies:**
```
#pragma omp parallel for if (input->rows > 1000)
  for (int i = 0; i < input->rows; i++) {
    activations[0]->data[i][0] = input->data[i][0];
  }
```

These are parallelized using the basic `for pragma`

##### Training loops


**Training loop: Parallelized stochastic gradient descent**

It is infact possible to parallelize the entire training loop, as in running multiple input images at a time, and reconciling the weights. [^1]
- Minor overwrites or "racing" between updates average out across many samples and epochs.
- Neural networks are **highly redundant** and tolerant to noise in updates.

```
#pragma omp parallel for reduction(+ : epoch_loss, correct)                    \
    schedule(dynamic, 100)
    for (int sample = 0; sample < batch_size; sample++) {
		// loop logic here
	}
```

- Each Thread iterates over a collection of samples
- **reduction** is used to make sure that `epoch_loss` and `correct` (correct stores count of correct guesses) are safely summed across threads
- `schedule(dynamic, 100)` is used to help with load balancing, each thread takes 100 samples at a time


**Forward pass**

```
copy_Mat(thread_activations[tid][0], inputs[sample]);
forward_pass(model, thread_activations[tid][0], thread_activations[tid]);
```
- Each thread stores activations on their own private buffers to prevent over writing

Dot product used for forward pass is parallelized through tiling as described above

**Backward pass**


```
backward_pass(model, thread_activations[tid], targets[sample]);
```
- Nothing inside the backward_pass() itself is parallelized
- Each thread runs the backward_pass() and updates gradients directly to the shared model
- Does not use locks for weights, knows as `Hogwild Stochastic Gradient Descent`
- Race conditions do occur but are statistically tolerable **IN SPECIFIC CASES!!**


##### Testing

This test was done using `random-nn-test.c`, this initializes a random dataset, each sample is size `1024`.

using this architecture
```
#define NUM_LAYERS 5
const int layer_sizes[NUM_LAYERS + 1] = {1024, 512, 256,
                                         architechture128,  64,  10}; // 1K input, 10 output
```

All hidden layers use **ReLU**, output layer uses **SoftMax**

**parallel 16 threads:**
```
=== Neural Network Speed Benchmark ===
Network Architecture: 1024 512 256 128 64 10
Generating 10000 random samples...
Starting training benchmark (5 epochs)...
Epoch 1/5 | Time: 43.68s | Loss: 2.3033 | Acc: 9.87%
Epoch 2/5 | Time: 44.42s | Loss: 2.3022 | Acc: 10.76%
Epoch 3/5 | Time: 44.33s | Loss: 2.3016 | Acc: 10.86%
Epoch 4/5 | Time: 44.71s | Loss: 2.3012 | Acc: 11.29%
Epoch 5/5 | Time: 44.43s | Loss: 2.3009 | Acc: 11.25%
Total training time: 221.59 seconds

=== Benchmark Results ===
Total time: 221.59 seconds
Time per epoch: 44.32 seconds
Samples per second: 226
```

**single threaded no parallel:**
```
=== Neural Network Speed Benchmark ===
Network Architecture: 1024 512 256 128 64 10
Generating 10000 random samples...
using openmp
Starting training benchmark (5 epochs)...
Epoch 1/5 | Time: 218.82s | Loss: 2.3028 | Acc: 9.92%
Epoch 2/5 | Time: 218.19s | Loss: 2.3023 | Acc: 10.07%
Epoch 3/5 | Time: 218.12s | Loss: 2.3019 | Acc: 10.60%
Epoch 4/5 | Time: 217.55s | Loss: 2.3016 | Acc: 10.72%
Epoch 5/5 | Time: 217.37s | Loss: 2.3013 | Acc: 10.80%
Total training time: 1090.05 seconds

=== Benchmark Results ===
Total time: 1090.05 seconds
Time per epoch: 218.01 seconds
Samples per second: 46
```

**Conclusion**

The dataset is randomized with no clear patterns, therefore accuracy and loss will be ignored. Additionally we are testing for speed and efficiency therefore we only care about the speed in this case.

single threading was ~$5\times$ slower than $16$ threads, with 16 threads running at $226$ SPS (Samples per second). Compared to the 46 SPS of the single threading application.


[^1]: HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent
	https://arxiv.org/abs/1106.5730
