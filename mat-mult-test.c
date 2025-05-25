
#include <math.h> // For fabsf()
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function prototypes
float **create_2d_matrix(int rows, int cols);
float *create_1d_matrix(int rows, int cols);
void free_2d_matrix(float **matrix, int rows);
void multiply_2d_matrices(int n, float **a, float **b, float **c);
void multiply_1d_matrices(int n, float *a, float *b, float *c);
void initialize_random_matrix(int n, float **matrix_2d, float *matrix_1d);
int compare_results(int n, float **c_2d, float *c_1d);

int main() {
  const int n = 2048;             // Matrix size (n x n)
  const float tolerance = 0.001f; // Tolerance for floating point comparison

  double start = 0;
  double end = 0;

  // Create and initialize matrices
  printf("Creating matrices of size %dx%d...\n", n, n);

  // 2D matrices
  float **a_2d = create_2d_matrix(n, n);
  float **b_2d = create_2d_matrix(n, n);
  float **c_2d = create_2d_matrix(n, n);

  // 1D matrices (contiguous)
  float *a_1d = create_1d_matrix(n, n);
  float *b_1d = create_1d_matrix(n, n);
  float *c_1d = create_1d_matrix(n, n);

  // Initialize with random values
  printf("Initializing matrices with random values...\n");
  initialize_random_matrix(n, a_2d, a_1d);
  initialize_random_matrix(n, b_2d, b_1d);

  // Verify initialization
  printf("Verifying initialization...\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (fabsf(a_2d[i][j] - a_1d[i * n + j]) > tolerance) {
        printf("Initialization mismatch at (%d,%d): 2D=%f, 1D=%f\n", i, j,
               a_2d[i][j], a_1d[i * n + j]);
      }
      if (fabsf(b_2d[i][j] - b_1d[i * n + j]) > tolerance) {
        printf("Initialization mismatch at (%d,%d): 2D=%f, 1D=%f\n", i, j,
               b_2d[i][j], b_1d[i * n + j]);
      }
    }
  }

  // Time 2D matrix multiplication
  printf("\nPerforming 2D matrix multiplication...\n");
  start = omp_get_wtime();
  multiply_2d_matrices(n, a_2d, b_2d, c_2d);
  end = omp_get_wtime();
  double time_taken = end - start;
  printf("2D multiplication took %.4f seconds\n", time_taken);

  // Time 1D matrix multiplication
  printf("\nPerforming 1D matrix multiplication...\n");
  start = omp_get_wtime();
  multiply_1d_matrices(n, a_1d, b_1d, c_1d);
  end = omp_get_wtime();
  time_taken = end - start;
  printf("1D multiplication took %.4f seconds\n", time_taken);

  // Compare results
  printf("\nComparing results...\n");
  int mismatch_count = compare_results(n, c_2d, c_1d);
  if (mismatch_count == 0) {
    printf("Results match perfectly!\n");
  } else {
    printf("Found %d mismatches (tolerance = %e)\n", mismatch_count, tolerance);
  }

  // Free memory
  free_2d_matrix(a_2d, n);
  free_2d_matrix(b_2d, n);
  free_2d_matrix(c_2d, n);
  free(a_1d);
  free(b_1d);
  free(c_1d);

  return 0;
}

// Compare results between 2D and 1D implementations
int compare_results(int n, float **c_2d, float *c_1d) {
  const float tolerance = 1e-2f;
  int mismatch_count = 0;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float diff = fabsf(c_2d[i][j] - c_1d[i * n + j]);
      if (diff > tolerance) {
        if (mismatch_count < 10) { // Print first 10 mismatches
          printf("Mismatch at (%d,%d): 2D=%f, 1D=%f, diff=%e\n", i, j,
                 c_2d[i][j], c_1d[i * n + j], diff);
        }
        mismatch_count++;
      }
    }
  }
  return mismatch_count;
}


// Create a 2D matrix (non-contiguous memory)
float **create_2d_matrix(int rows, int cols) {
  float **matrix = (float **)malloc(rows * sizeof(float *));
  if (matrix == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < rows; i++) {
    matrix[i] = (float *)malloc(cols * sizeof(float));
    if (matrix[i] == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(EXIT_FAILURE);
    }
  }

  return matrix;
}

// Create a 1D matrix (contiguous memory)
float *create_1d_matrix(int rows, int cols) {
  float *matrix = (float *)malloc(rows * cols * sizeof(float));
  if (matrix == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  return matrix;
}

// Free 2D matrix memory
void free_2d_matrix(float **matrix, int rows) {
  for (int i = 0; i < rows; i++) {
    free(matrix[i]);
  }
  free(matrix);
}

// 2D matrix multiplication
void multiply_2d_matrices(int n, float **a, float **b, float **c) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      c[i][j] = 0;
      for (int k = 0; k < n; k++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

// 1D matrix multiplication (row-major order)
void multiply_1d_matrices(int n, float *a, float *b, float *c) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < n; k++) {
      float a_ik = a[i * n + k];
      for (int j = 0; j < n; j++) {
        c[i * n + j] += a_ik * b[k * n + j];
      }
    }
  }
}

// Initialize both matrix types with the same random values
void initialize_random_matrix(int n, float **matrix_2d, float *matrix_1d) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float val = (float)rand() / RAND_MAX; // Random float between 0 and 1
      matrix_2d[i][j] = val;
      matrix_1d[i * n + j] = val;
    }
  }
}
