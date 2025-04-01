
#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define UPPER_BOUND 100
#define LOWER_BOUND 1

typedef struct {
  int rows, columns;
  int **data;
} Matrix;

/*
  allocate a matrix of size rows x columns
  returns a Struct Matrix
*/
Matrix *init_Matrix(int rows, int columns);

void matrix_Free(Matrix *mat);

/*
  randomize matrix of size rows x columns
  values are randomized between values LOWER_BOUND and UPPER_BOUND
*/
void randomize_Matrix(Matrix *matrix);

void print_Matrix(Matrix *matrix);

/*
  add 2 matricies of equal dimensions
  exits with code 1 if dimensions unequal
  return Matrix struct result
*/
Matrix *add_Mat(Matrix *mat1, Matrix *mat2);

Matrix *dot_Mat(Matrix *mat1, Matrix *mat2);

#endif /* MATRIX_H */
