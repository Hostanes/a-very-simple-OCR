#include "lib/matrix-math.h"

int main() {

  srand(time(0));

  int m_rows = 5;
  int m_cols = 10;

  Matrix *mat1 = init_Matrix(m_rows, m_cols);
  Matrix *mat2 = init_Matrix(m_rows, m_cols);

  int v_rows = m_cols;
  Matrix *vec1 = init_Matrix(v_rows, 1);

  randomize_Matrix(mat2);
  randomize_Matrix(mat1);
  randomize_Matrix(vec1);

  print_Matrix(mat1);
  print_Matrix(vec1);

  Matrix *add_result = add_Mat(mat1, mat2);

  print_Matrix(add_result);
}
