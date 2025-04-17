
## What was parallelized

There are 2 main methods used when parallelizing neural networks:
- Parallelizing "**over Propagation**"
- Parallelizing "**over Samples**"

#### Over propagation:

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
For a comparison of both methods see the standalone C script`tiling-vs-naive.c`

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

#### Over Samples:

Instead of parallelizing each network operation over a single image, we spread out the input images over each thread, and reconcile the different weight updates, this reconciliation can be done in various ways:

**Mutex locks**
use a mutex lock for each weight index, this ensures no race conditions and each thread waits its turn to update. However in conditions where there are frequent updates over the network this would negate the benefits of parallelization.

**Batching and accumulating**

A parallel region where each thread takes a batch of images at a time example `batch_Size = 32`, each thread runs and stores the gradient on its own locally stored memory. This is followed by a  serial `critical` region that reconciles each threads private accumulated weights.

First we divide the samples over batches for each thread, for example each thread takes `batch_size = 16` images at a time. Each thread runs the forward passes over the images and computes their local gradients.
```
// Parallel minibatch processing
#pragma omp for schedule(dynamic)
      for (int b = 0; b < train_size; b += batch_size) {
        float batch_loss = 0;

        // Process minibatch
        for (int i = b; i < b + batch_size && i < train_size; i++) {
          float img[INPUT_SIZE];
          float target[OUTPUT_SIZE];

          normalize_images(&data.images[i * INPUT_SIZE], img, 1);
          memset(target, 0, sizeof(target));
          target[data.labels[i]] = 1.0f;

          compute_gradients(net, img, target, thread_gradients,
                            thread_bias_gradients);

          float *output = net->layers[net->num_layers - 1].output;
          batch_loss += -logf(output[data.labels[i]] + 1e-10f);
        }

```

This is followed by the critical region which applies the updates using all the private per-thread gradients.
```
// Synchronized weight update, TODO this is the largest bottleneck
#pragma omp critical
        {
          apply_updates(net, thread_gradients, thread_bias_gradients,
                        batch_size);

          // Reset gradients
          for (int l = 0; l < net->num_layers; l++) {
            memset(thread_gradients[l], 0,
                   net->layers[l].input_size * net->layers[l].output_size *
                       sizeof(float));
            memset(thread_bias_gradients[l], 0,
                   net->layers[l].output_size * sizeof(float));
          }
        } // END critical
```

