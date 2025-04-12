/*
  ut-matrix-math.c

  unit test of the matrix math library, tests out:
  - initializing matrix
  - adding matricies
  - multiplying matricies (dot product)
*/


#include "lib/matrix-math.h"

int main() {

  srand(time(0));

  int m_rows = 5;
  int m_cols = 10;

  Matrix *mat1 = init_Matrix(m_rows, m_cols);
  Matrix *mat2 = init_Matrix(m_rows, m_cols);
  Matrix *mat3 = init_Matrix(m_cols, 2);

  int v_rows = m_cols;
  Matrix *vec1 = init_Matrix(v_rows, 1);

  randomize_Matrix(mat1);
  randomize_Matrix(mat2);
  randomize_Matrix(mat3);
  randomize_Matrix(vec1);

  printf("matrix 1 : \n");
  print_Matrix(mat1);
  printf("matrix 2 : \n");
  print_Matrix(mat2);
  printf("matrix 3 : \n");
  print_Matrix(mat3);
  printf("vector 1 : \n");
  print_Matrix(vec1);

  Matrix *add_result = add_Mat(mat1, mat2);

  printf("result of a matrix addition:\n");
  print_Matrix(add_result);

  Matrix *dot_result = dot_Mat(mat1, mat3);
  printf("result of matrix dot product:\n");
  print_Matrix(dot_result);

  return 0;
}
