#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>

#include "mnist_file.h"
#include "neural_network.h"

#define STEPS 1000
#define BATCH_SIZE 100

const char * train_images_file = "../data/train-images-idx3-ubyte";
const char * train_labels_file = "../data/train-labels-idx1-ubyte";
const char * test_images_file  = "../data/train-images-idx3-ubyte";
const char * test_labels_file  = "../data/train-labels-idx1-ubyte";

float calculate_accuracy(mnist_dataset_t * dataset, neural_network_t * network) {
    float activations[MNIST_LABELS], max_activation;
    int i, j, correct = 0, predict;

    for (i = 0; i < dataset->size; i++) {
        neural_network_hypothesis(&dataset->images[i], network, activations);
        for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
            if (max_activation < activations[j]) {
                max_activation = activations[j];
                predict = j;
            }
        }
        if (predict == dataset->labels[i]) correct++;
    }

    return ((float) correct) / ((float) dataset->size);
}

int main(int argc, char *argv[]) {
    mnist_dataset_t *train_dataset, *test_dataset;
    mnist_dataset_t batch;
    neural_network_t network;
    float local_loss, global_loss, accuracy;
    int i, rank, size, batches;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    train_dataset = mnist_get_dataset(train_images_file, train_labels_file);
    test_dataset  = mnist_get_dataset(test_images_file, test_labels_file);

    if (rank == 0) {
        neural_network_random_weights(&network);
    }

    // Broadcast the initial weights to all processes
    MPI_Bcast(&network, sizeof(neural_network_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    batches = train_dataset->size / BATCH_SIZE;

    for (i = 0; i < STEPS; i++) {
        int batch_index = (i * size + rank) % batches;

        // Get a specific batch per process
        mnist_batch(train_dataset, &batch, BATCH_SIZE, batch_index);

        // Each process performs a training step on its own batch
        local_loss = neural_network_training_step(&batch, &network, 0.5);

        // Reduce (average) the loss across all processes
        MPI_Allreduce(&local_loss, &global_loss, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        global_loss /= size;

        // Broadcast updated weights/biases from rank 0 to sync all networks
        MPI_Bcast(&network, sizeof(neural_network_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Only one process calculates accuracy and prints
        if (rank == 0) {
            accuracy = calculate_accuracy(test_dataset, &network);
            printf("Step %04d\tAvg Loss: %.2f\tAccuracy: %.3f\n", i, global_loss / BATCH_SIZE, accuracy);
        }
    }

    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);
    MPI_Finalize();
    return 0;
}
