```
  A dense neural network library attempting to parallelize
  the operations on the CPU using OpenMP and GPU using OpenCL
  in a memory efficient manner
```

# Parallelized DNN operations


the current architechture uses a dense neural network of custom user defined sizes, all hidden layers use ReLU and the final layer uses softmax. Plans are to expand this to use custom user defined functions using function hooks
