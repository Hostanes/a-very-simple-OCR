#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define UPPER_BOUND 100
#define LOWER_BOUND 1

void add_Vector_Vector() {}

/*
  randomize matrix of size rows x columns
  values are randomized between values LOWER_BOUND and UPPER_BOUND
*/
void randomize_Matrix(int rows, int columns, int matrix[rows][columns]) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      matrix[i][j] = rand() % (UPPER_BOUND - LOWER_BOUND + 1) + LOWER_BOUND;
    }
  }
}

void print_Matrix(int rows, int columns, int matrix[rows][columns]) {
  printf("--\n");
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      printf("%d, ", matrix[i][j]);
    }
    printf("\n");
  }
  printf("--\n");
}

int main() {

  srand(time(0));

  int m_rows = 5;
  int m_cols = 10;
  int matrix[m_rows][m_cols];

  int v_cols = m_cols;
  int vector[v_cols][1];

  randomize_Matrix(m_rows, m_cols, matrix);
  print_Matrix(m_rows, m_cols, matrix);

  printf("\n");

  randomize_Matrix(v_cols, 1, vector);
  print_Matrix(v_cols, 1, vector);
}
