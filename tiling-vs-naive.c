
/*
  dot-prod-test.c

  compares 2 methods to parallelize matrix dot product/ matrix multiplication
  1. naive_matmul():  Naive parallel, simply parallelizing the outer for loop
  2. blocked_matmul(): divides matricies into blocks for better cache effiency

  can use perf utility to compare cache misses using this command:
    $ perf stat -e cache-misses,L1-dcache-load-misses ./test.o
  where test.o is the compiled binary of this script
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4096
double A[N][N], B[N][N], C[N][N];

void init_matrices() {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = (double)rand() / RAND_MAX;
      B[i][j] = (double)rand() / RAND_MAX;
      C[i][j] = 0.0;
    }
  }
}

void naive_matmul() {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double sum = 0.0;
      for (int k = 0; k < N; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

void blocked_matmul(int BLOCK_SIZE) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i += BLOCK_SIZE) {
    for (int j = 0; j < N; j += BLOCK_SIZE) {
      for (int k = 0; k < N; k += BLOCK_SIZE) {
        for (int ii = i; ii < i + BLOCK_SIZE && ii < N; ii++) {
          for (int jj = j; jj < j + BLOCK_SIZE && jj < N; jj++) {
            double sum = 0.0;
            for (int kk = k; kk < k + BLOCK_SIZE && kk < N; kk++) {
              sum += A[ii][kk] * B[kk][jj];
            }
#pragma omp atomic
            C[ii][jj] += sum;
          }
        }
      }
    }
  }
}

int main() {
  init_matrices();

  double start, end;
  int num_threads = omp_get_max_threads();
  printf("Running with %d threads\n", num_threads);

  // start = omp_get_wtime();
  // naive_matmul();
  // end = omp_get_wtime();
  // printf("Naive parallel: %f sec\n", end - start);

  // #pragma omp parallel for collapse(2)
  //   for (int i = 0; i < N; i++) {
  //     for (int j = 0; j < N; j++) {
  //       C[i][j] = 0.0;
  //     }
  //   }

  start = omp_get_wtime();
  blocked_matmul(128);
  end = omp_get_wtime();
  printf("Blocked parallel: %f sec\n", end - start);

  return 0;
}
