#include <cstring>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <ctime>

using namespace std;
#define NMAX_FEATURES 255

// Define the characteristics of a sample placed in r^n
struct Sample {
    double features[NMAX_FEATURES]; // Sample coordinates
    int _class;                     // Group of sample
    double distance;                // Distance from a single specific test sample
};

// Used to sort an array of samples by ascending order of distance
bool comparison(Sample a, Sample b) { return (a.distance < b.distance); }

// Get the euclidian distance between two samples placed in r^n
double euclidian_distance(Sample train_sample, Sample test_sample, int N_FEATURES) {
    double sum = 0;
    for (int i = 0; i < N_FEATURES; i++) {
        sum += pow((train_sample.features[i] - test_sample.features[i]), 2);
    }
    return sqrt(sum);
}

// Find the class of a test sample using K nearest neighbor algorithm
int knn_prediction(Sample train_samples[], int N_TRAIN, int K, Sample test_sample, int N_CLASSES, int N_FEATURES) {
    for (int i = 0; i < N_TRAIN; i++)
        train_samples[i].distance = euclidian_distance(train_samples[i], test_sample, N_FEATURES);

    sort(train_samples, train_samples + N_TRAIN, comparison);

    int freq[N_CLASSES];
    for (int i = 0; i < N_CLASSES; i++) freq[i] = -1;
    for (int i = 0; i < K; i++) freq[train_samples[i]._class] += 1;

    int max = -1, predicted_class = -1;
    for (int i = 0; i < N_CLASSES; i++) {
        if (freq[i] > max) {
            max = freq[i];
            predicted_class = i;
        }
    }
    return predicted_class;
}

// Return the accuracy of the model
double get_accuracy(Sample train_samples[], Sample test_samples[], int N_TRAIN, int N_TEST, int K, int N_CLASSES, int N_FEATURES) {
    int correctly_classified_samples = 0;
    for (int i = 0; i < N_TEST; i++) {
        if (test_samples[i]._class == knn_prediction(train_samples, N_TRAIN, K, test_samples[i], N_CLASSES, N_FEATURES))
            correctly_classified_samples += 1;
    }
    return (correctly_classified_samples / (double)N_TEST) * 100;
}

// Import data samples from a given file
void read_dataset(ifstream& file, Sample dataset_samples[], int N_SAMPLES, int N_FEATURES) {
    file.seekg(0, ios::beg);
    for (int i = 0; i <= N_SAMPLES; i++) {
        if (file.eof()) break;
        for (int j = 0; j < N_FEATURES; j++) file >> dataset_samples[i].features[j];
        file >> dataset_samples[i]._class;
    }
    file.close();
}

// Main program
int main(int argc, char** argv) {
    clock_t start_time = clock();
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        cout << "Usage: <train_file> <N_TRAIN> <test_file> <N_TEST> <K> <N_FEATURES> <N_CLASSES>\n";
        cout << "Example: dataset/train.txt 100 dataset/test.txt 40 3 2 2\n";
        return EXIT_SUCCESS;
    }

    if (argc < 8) {
        cerr << "Error: Please insert all the necessary parameters (use --help for description).\n";
        return EXIT_FAILURE;
    }

    ifstream train_file(argv[1]);
    ifstream test_file(argv[3]);

    if (!train_file || !test_file) {
        cerr << "Error: Could not open input files.\n";
        return EXIT_FAILURE;
    }

    int N_TRAIN = stoi(argv[2]);
    int N_TEST = stoi(argv[4]);
    int K = stoi(argv[5]);
    int N_FEATURES = stoi(argv[6]);
    int N_CLASSES = stoi(argv[7]);

    if (N_TRAIN <= 0 || N_TEST <= 0 || K <= 0 || N_FEATURES <= 0 || N_CLASSES <= 0) {
        cerr << "Error: All numeric parameters must be positive.\n";
        return EXIT_FAILURE;
    }

    if (N_FEATURES > NMAX_FEATURES) {
        cerr << "Error: Maximum number of features is " << NMAX_FEATURES << ".\n";
        return EXIT_FAILURE;
    }

    Sample* train_samples = new Sample[N_TRAIN];
    read_dataset(train_file, train_samples, N_TRAIN, N_FEATURES);

    Sample* test_samples = new Sample[N_TEST];
    read_dataset(test_file, test_samples, N_TEST, N_FEATURES);

    printf("%dNN Train accuracy: %.2f%%\n", K, get_accuracy(train_samples, train_samples, N_TRAIN, N_TRAIN, K, N_CLASSES, N_FEATURES));
    printf("%dNN Test accuracy: %.2f%%\n", K, get_accuracy(train_samples, test_samples, N_TRAIN, N_TEST, K, N_CLASSES, N_FEATURES));
    clock_t end_time = clock();
    double elapsed_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Execution time: %.3fs (%d samples)\n", elapsed_time, N_TRAIN + N_TEST);
    delete[] train_samples;
    delete[] test_samples;

    return EXIT_SUCCESS;
}
