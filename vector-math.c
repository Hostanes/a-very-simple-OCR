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
  randomize matrix of size rows x columns
  values are randomized between values LOWER_BOUND and UPPER_BOUND
*/
void randomize_Matrix(Matrix matrix) {
  for (int i = 0; i < matrix.rows; i++) {
    for (int j = 0; j < matrix.columns; j++) {
      matrix.data[i][j] =
          rand() % (UPPER_BOUND - LOWER_BOUND + 1) + LOWER_BOUND;
    }
  }
}

void print_Matrix(Matrix matrix) {
  printf("--\n");
  for (int i = 0; i < matrix.rows; i++) {
    for (int j = 0; j < matrix.columns; j++) {
      printf("%d, ", matrix.data[i][j]);
    }
    printf("\n");
  }
  printf("--\n");
}

/*
  allocate a matrix of size rows x columns
  returns a Struct Matrix
*/
Matrix matrix_Init(int rows, int columns) {
  Matrix mat;
  mat.rows = rows;
  mat.columns = columns;

  mat.data = (int **)malloc(rows * sizeof(int *));
  if (!mat.data) {
    fprintf(stderr, "matrix malloc failed\n");
    exit(1);
  }

  for (int i = 0; i < rows; i++) {
    mat.data[i] = (int *)malloc(columns * sizeof(int));
    if (!mat.data[i]) {
      fprintf(stderr, "matrix malloc failed\n");
      exit(1);
    }

    for (int j = 0; j < columns; j++) {
      mat.data[i][j] = 0;
    }
  }

  return mat;
}

void matrix_Free(Matrix mat) {
  for (int i = 0; i < mat.rows; i++) {
    free(mat.data[i]);
  }
  free(mat.data);
}

int main() {

  srand(time(0));

  int m_rows = 5;
  int m_cols = 10;
  Matrix mat1 = matrix_Init(m_rows, m_cols);

  int v_rows = m_cols;
  Matrix vec1 = matrix_Init(v_rows, 1);

  randomize_Matrix(mat1);
  randomize_Matrix(vec1);

  print_Matrix(mat1);
  print_Matrix(vec1);
}
