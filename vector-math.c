#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define UPPER_BOUND 100
#define LOWER_BOUND 1







/*
  randomize matrix of size rows x columns
  values are randomized between values LOWER_BOUND and UPPER_BOUND
*/
void randomize_matrix(int rows, int columns, int matrix[rows][columns]) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      matrix[i][j] = rand() % (UPPER_BOUND - LOWER_BOUND + 1) + LOWER_BOUND;
    }
  }
}

/*
  randomize vector of length `size`
  values are randomized between values LOWER_BOUND and UPPER_BOUND
*/
void randomize_vector(int size, int matrix[size]) {
  for (int i = 0; i < size; i++) {
    matrix[i] = rand() % (UPPER_BOUND - LOWER_BOUND + 1) + LOWER_BOUND;
  }
}

void print_matrix(int rows, int columns, int matrix[rows][columns]) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      printf("%d, ", matrix[i][j]);
    }
    printf("\n");
  }
}

void print_vector(int size, int vector[size]) {
  for (int i = 0; i < size; i++) {
    printf("%d\n", vector[i]);
  }
}

int main() {

  srand(time(0));

  int m_rows = 5;
  int m_cols = 10;
  int matrix[m_rows][m_cols];

  int v_cols = m_cols;
  int vector[v_cols];

  randomize_matrix(m_rows, m_cols, matrix);
  print_matrix(m_rows, m_cols, matrix);

  printf("\n");

  randomize_vector(m_rows, vector);
  print_vector(m_rows, vector);
}
