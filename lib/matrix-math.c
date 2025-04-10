#include "matrix-math.h"

/*
  Adds 2 matricies of equal dimensions
  mat1: m x n matrix
  mat2: k x p matrix
  Returns: pointer to Matrix struct result
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

/*
  Performs element wise multiplication for 2 matricies of equal dimensions
  mat1: m x n matrix
  mat2: k x p matrix
  Returns: pointer to Matrix struct result
*/
Matrix *multiply_Mat(Matrix *mat1, Matrix *mat2) {

  if (mat1->rows != mat2->rows || mat1->columns != mat2->columns) {
    fprintf(stderr, "ERROR: matrix add: matrix dimensions dont match");
    exit(1);
  }
  Matrix *result = init_Matrix(mat1->rows, mat1->columns);
  for (int rows = 0; rows < mat1->rows; rows++) {
    for (int columns = 0; columns < mat1->columns; columns++) {
      result->data[rows][columns] =
          mat1->data[rows][columns] * mat2->data[rows][columns];
    }
  }
  return result;
}

/*
   Performs matrix multiplication (dot product) of two matrices
   mat1: m x n matrix (left operand)
   mat2: n x p matrix (right operand)
   Returns: Pointer to new m x p matrix (must be freed by caller)
*/
Matrix *dot_Mat(Matrix *mat1, Matrix *mat2) {
  if (mat1->columns != mat2->rows) {
    fprintf(stderr,
            "ERROR: Dot Product: inner dimensions don't match (%d != %d)\n",
            mat1->columns, mat2->rows);
    exit(1);
  }

  Matrix *result = init_Matrix(mat1->rows, mat2->columns);

  for (int i = 0; i < mat1->rows; i++) {
    for (int j = 0; j < mat2->columns; j++) {
      double sum = 0;
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
      matrix->data[i][j] = (double )rand() / RAND_MAX * 1;
    }
  }
}

void print_Matrix(Matrix *matrix) {
  printf("--\n");
  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->columns; j++) {
      printf("%.2f, ", matrix->data[i][j]);
    }
    printf("\n");
  }
  printf("--\n");
}

/*
  Allocate a matrix of size rows x columns
  rows: number of rows (each row is an array of size columns)
  columns: number of columns
  Returns: a pointer to Struct Matrix with all 0.0 data
*/
Matrix *init_Matrix(int rows, int columns) {
  Matrix *mat;
  mat = malloc(sizeof(Matrix));
  mat->rows = rows;
  mat->columns = columns;

  mat->data = (double **)malloc(rows * sizeof(double *));
  if (!mat->data) {
    fprintf(stderr, "ERROR: matrix malloc failed\n");
    exit(1);
  }

  for (int i = 0; i < rows; i++) {
    mat->data[i] = (double *)malloc(columns * sizeof(double));
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
