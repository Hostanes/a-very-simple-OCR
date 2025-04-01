
#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define UPPER_BOUND 100
#define LOWER_BOUND 1

typedef struct {
  int rows, columns;
  double **data;
} Matrix;

/*
  Allocate a matrix of size rows x columns
  rows: number of rows (each row is an array of size columns)
  columns: number of columns
  Returns: a pointer to Struct Matrix with all 0.0 data
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
  Adds 2 matricies of equal dimensions
  mat1: m x n matrix
  mat2: k x p matrix
  Returns: pointer to Matrix struct result
*/
Matrix *add_Mat(Matrix *mat1, Matrix *mat2);

/*
   Performs matrix multiplication (dot product) of two matrices
   mat1: m x n matrix (left operand)
   mat2: n x p matrix (right operand)
   Returns: Pointer to new m x p matrix (must be freed by caller)
*/
Matrix *dot_Mat(Matrix *mat1, Matrix *mat2);

#endif /* MATRIX_H */
