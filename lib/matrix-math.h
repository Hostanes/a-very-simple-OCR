
#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#include <math.h>
#include <stdlib.h>

void matrix_dot(float *A, float *B, float *C, int m, int n, int p);
void matrix_add(float *A, float *B, int size);
void matrix_scale(float *A, float scalar, int size);
void matrix_softmax(float *input, int size);
void matrix_relu(float *input, int size);
void matrix_relu_derivative(float *input, float *output, int size);

#endif
