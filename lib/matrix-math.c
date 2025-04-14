
#include "matrix-math.h"

void matrix_dot(float *A, float *B, float *C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i*p + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i*p + j] += A[i*n + k] * B[k*p + j];
            }
        }
    }
}

void matrix_add(float *A, float *B, int size) {
    for (int i = 0; i < size; i++) {
        A[i] += B[i];
    }
}

void matrix_scale(float *A, float scalar, int size) {
    for (int i = 0; i < size; i++) {
        A[i] *= scalar;
    }
}

void matrix_softmax(float *input, int size) {
    float max = input[0], sum = 0;
    for (int i = 1; i < size; i++)
        if (input[i] > max) max = input[i];
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    for (int i = 0; i < size; i++)
        input[i] /= sum;
}

void matrix_relu(float *input, int size) {
    for (int i = 0; i < size; i++) {
        input[i] = input[i] > 0 ? input[i] : 0;
    }
}

void matrix_relu_derivative(float *input, float *output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] > 0 ? 1 : 0;
    }
}
