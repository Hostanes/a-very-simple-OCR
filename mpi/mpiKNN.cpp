
#include <cstring>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <mpi.h>

using namespace std;
#define NMAX_FEATURES 255

struct Sample {
    double features[NMAX_FEATURES];
    int _class;
    double distance;
};

bool comparison(Sample a, Sample b) {
    return (a.distance < b.distance);
}

double euclidian_distance(Sample train_sample, Sample test_sample, int N_FEATURES) {
    double sum = 0;
    for (int i = 0; i < N_FEATURES; i++)
        sum += pow((train_sample.features[i] - test_sample.features[i]), 2);
    return sqrt(sum);
}

int knn_prediction(Sample train_samples[], int N_TRAIN, int K, Sample test_sample, int N_CLASSES, int N_FEATURES) {
    for (int i = 0; i < N_TRAIN; i++)
        train_samples[i].distance = euclidian_distance(train_samples[i], test_sample, N_FEATURES);

    sort(train_samples, train_samples + N_TRAIN, comparison);

    int freq[N_CLASSES];
    for (int i = 0; i < N_CLASSES; i++) freq[i] = -1;
    for (int i = 0; i < K; i++) freq[train_samples[i]._class] += 1;

    int max = -1, predicted_class = -1;
    for (int i = 0; i < N_CLASSES; i++) {
        if ((freq[i] > max)) {
            max = freq[i];
            predicted_class = i;
        }
    }
    return predicted_class;
}

double get_accuracy(Sample train_samples[], Sample test_samples[], int N_TRAIN, int N_TEST, int K, int N_CLASSES, int N_FEATURES) {
    int correctly_classified_samples = 0;
    for (int i = 0; i < N_TEST; i++) {
        if (test_samples[i]._class == knn_prediction(train_samples, N_TRAIN, K, test_samples[i], N_CLASSES, N_FEATURES))
            correctly_classified_samples += 1;
    }
    return (correctly_classified_samples / (double)N_TEST) * 100;
}

void read_dataset(ifstream &file, Sample dataset_samples[], int N_SAMPLES, int N_FEATURES) {
    file.seekg(0, ios::beg);
    for (int i = 0; i <= N_SAMPLES; i++) {
        if (file.eof()) break;
        for (int j = 0; j < N_FEATURES; j++) file >> dataset_samples[i].features[j];
        file >> dataset_samples[i]._class;
    }
    file.close();
}

int main(int argc, char** argv) {
    int MYRANK, SIZE;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &MYRANK);
    MPI_Comm_size(MPI_COMM_WORLD, &SIZE);
    double startTime = MPI_Wtime();

    if ((argc == 2 && strcmp(argv[1], "--help") == 0)) {
        if (MYRANK == 0)
            printf("Usage: mpirun -np <num_procs> ./knn_parallel <train_file> <N_TRAIN> <test_file> <N_TEST> <K> <N_FEATURES> <N_CLASSES>\n");
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    if (argc < 8) {
        if (MYRANK == 0)
            fprintf(stderr, "Error: Missing parameters. Use --help for usage.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    ifstream train_file(argv[1]);
    ifstream test_file(argv[3]);
    int N_TRAIN = stoi(argv[2]);
    int N_TEST = stoi(argv[4]);
    int K = stoi(argv[5]);
    int N_FEATURES = stoi(argv[6]);
    int N_CLASSES = stoi(argv[7]);

    Sample* train_samples = new Sample[N_TRAIN];
    Sample* test_samples = new Sample[N_TEST];

    if (MYRANK == 0) {
        if (!train_file || !test_file) {
            fprintf(stderr, "Error: Failed to open dataset files.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (N_TRAIN <= 0 || N_TEST <= 0 || K <= 0 || N_FEATURES <= 0 || N_CLASSES <= 0) {
            fprintf(stderr, "Error: All numeric arguments must be positive.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (N_FEATURES > NMAX_FEATURES) {
            fprintf(stderr, "Error: Maximum number of features is %d\n", NMAX_FEATURES);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        read_dataset(train_file, train_samples, N_TRAIN, N_FEATURES);
        read_dataset(test_file, test_samples, N_TEST, N_FEATURES);
    }

    MPI_Bcast(train_samples, sizeof(Sample) * N_TRAIN, MPI_BYTE, 0, MPI_COMM_WORLD);

    // TRAIN accuracy
    int n_processed_data = N_TRAIN / SIZE;
    int REMAINDER = N_TRAIN % SIZE;
    if (MYRANK == (SIZE - 1) && REMAINDER != 0) n_processed_data += REMAINDER;

    Sample* train_samples_node = new Sample[n_processed_data];
    for (int i = 0; i < n_processed_data; i++) {
        int idx = (MYRANK == SIZE - 1 && REMAINDER != 0) ?
                    i - REMAINDER + MYRANK * (N_TRAIN / SIZE) :
                    i + MYRANK * (N_TRAIN / SIZE);
        train_samples_node[i] = train_samples[idx];
    }

    double weighted_train_acc = get_accuracy(train_samples, train_samples_node, N_TRAIN, n_processed_data, K, N_CLASSES, N_FEATURES) * n_processed_data;
    double total_train_acc;
    MPI_Reduce(&weighted_train_acc, &total_train_acc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (MYRANK == 0)
        printf("%dNN Train accuracy: %.2f%%\n", K, total_train_acc / N_TRAIN);

    // TEST accuracy
    n_processed_data = N_TEST / SIZE;
    Sample* test_samples_node = new Sample[n_processed_data];
    MPI_Scatter(test_samples, n_processed_data * sizeof(Sample), MPI_BYTE,
                test_samples_node, n_processed_data * sizeof(Sample), MPI_BYTE, 0, MPI_COMM_WORLD);

    double node_test_acc = get_accuracy(train_samples, test_samples_node, N_TRAIN, n_processed_data, K, N_CLASSES, N_FEATURES);
    double sum_test_acc;
    MPI_Reduce(&node_test_acc, &sum_test_acc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    REMAINDER = N_TEST % SIZE;
    if (MYRANK == 0) {
        double weighted_remainder_acc = 0;
        if (REMAINDER != 0) {
            Sample* test_samples_rem = new Sample[REMAINDER];
            for (int i = 0; i < REMAINDER; i++)
                test_samples_rem[i] = test_samples[i + SIZE * n_processed_data];
            weighted_remainder_acc = get_accuracy(train_samples, test_samples_rem, N_TRAIN, REMAINDER, K, N_CLASSES, N_FEATURES) * REMAINDER;
            delete[] test_samples_rem;
        }
        double final_test_acc = ((sum_test_acc * n_processed_data) + weighted_remainder_acc) / N_TEST;
        printf("%dNN Test accuracy: %.2f%%\n", K, final_test_acc);

        double exec_time = MPI_Wtime() - startTime;
        printf("Execution time: %.3fs (%d samples, %d processes)\n", exec_time, N_TRAIN + N_TEST, SIZE);
    }

    delete[] train_samples;
    delete[] test_samples;
    delete[] train_samples_node;
    delete[] test_samples_node;

    MPI_Finalize();
    return 0;
}

