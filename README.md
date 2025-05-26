
```
  A parallelized implementation of a dense neural network
  comparing 2 methods
  1. Contiguous memory allocation, flat 1D arrays for weights, biases, values, gradients...
  2. Non contiguous, each layer stores its own information in a layer_t struct
```


# Contiguous vs non-Contiguous

### Scripts:

**How to run:**

`CnR.sh`: compiles the selected C script using its correspondig nnlib.c and runs it
    Example usage: `./CnR.sh ./contig/mnist-train-contig.c`

\[TODO]

