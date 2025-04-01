
#include "matrix-math.h"

/*
  add 2 matricies of equal dimensions
  exits with code 1 if dimensions unequal
  return Matrix struct result
*/
Matrix *add_Mat(Matrix *mat1, Matrix *mat2) {

  if (mat1->rows != mat2->rows || mat1->columns != mat2->columns) {
    fprintf(stderr, "ERROR: matrix add: matrix dimensions dont match");
    exit(1);
  }
  Matrix *result = init_Matrix(mat1->rows, mat1->columns);
  for (int rows = 0; rows < mat1->rows; rows++) {
    for (int columns = 0; columns < mat1->columns; columns++) {
      result->data[rows][columns] =
          mat1->data[rows][columns] + mat2->data[rows][columns];
    }
  }
  return result;
}

/**
 * Performs matrix multiplication (dot product) of two matrices
 * mat1: m x n matrix (left operand)
 * mat2: n x p matrix (right operand)
 * Returns: Pointer to new m x p matrix (must be freed by caller)
 */
Matrix *dot_Mat(Matrix *mat1, Matrix *mat2) {
  if (mat1->columns != mat2->rows) {
    fprintf(stderr,
            "ERROR: matrix multiply: inner dimensions don't match (%d != %d)\n",
            mat1->columns, mat2->rows);
    exit(1);
  }

  Matrix *result = init_Matrix(mat1->rows, mat2->columns);

  for (int i = 0; i < mat1->rows; i++) {
    for (int j = 0; j < mat2->columns; j++) {
      int sum = 0;
      for (int k = 0; k < mat1->columns; k++) {
        sum += mat1->data[i][k] * mat2->data[k][j];
      }
      result->data[i][j] = sum;
    }
  }

  return result;
}

/*
  randomize matrix of size rows x columns
  values are randomized between values LOWER_BOUND and UPPER_BOUND
*/
void randomize_Matrix(Matrix *matrix) {
  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->columns; j++) {
      matrix->data[i][j] =
          rand() % (UPPER_BOUND - LOWER_BOUND + 1) + LOWER_BOUND;
    }
  }
}

void print_Matrix(Matrix *matrix) {
  printf("--\n");
  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->columns; j++) {
      printf("%d, ", matrix->data[i][j]);
    }
    printf("\n");
  }
  printf("--\n");
}

/*
  allocate a matrix of size rows x columns
  returns a Struct Matrix
*/
Matrix *init_Matrix(int rows, int columns) {
  Matrix *mat;
  mat = malloc(sizeof(Matrix));
  mat->rows = rows;
  mat->columns = columns;

  mat->data = (int **)malloc(rows * sizeof(int *));
  if (!mat->data) {
    fprintf(stderr, "ERROR: matrix malloc failed\n");
    exit(1);
  }

  for (int i = 0; i < rows; i++) {
    mat->data[i] = (int *)malloc(columns * sizeof(int));
    if (!mat->data[i]) {
      fprintf(stderr, "ERROR: matrix malloc failed\n");
      exit(1);
    }

    for (int j = 0; j < columns; j++) {
      mat->data[i][j] = 0;
    }
  }

  return mat;
}

void matrix_Free(Matrix *mat) {
  for (int i = 0; i < mat->rows; i++) {
    free(mat->data[i]);
  }
  free(mat->data);
  free(mat);
}
