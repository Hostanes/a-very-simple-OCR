
// matmul.cl

__kernel void matmul_naive(__global const float *A, __global const float *B,
                           __global float *C, const int N) {
  int row = get_global_id(1);
  int col = get_global_id(0);
  float sum = 0.0f;

  for (int k = 0; k < N; ++k) {
    sum += A[row * N + k] * B[k * N + col];
  }

  C[row * N + col] = sum;
}

__kernel void matmul_tiled(__global const float *A, __global const float *B,
                           __global float *C, const int N) {
  const int BLOCK_SIZE = 16;

  int row = get_global_id(1);
  int col = get_global_id(0);

  int local_row = get_local_id(1);
  int local_col = get_local_id(0);

  __local float Asub[BLOCK_SIZE][BLOCK_SIZE];
  __local float Bsub[BLOCK_SIZE][BLOCK_SIZE];

  float sum = 0.0f;

  for (int bk = 0; bk < N / BLOCK_SIZE; ++bk) {
    Asub[local_row][local_col] = A[row * N + bk * BLOCK_SIZE + local_col];
    Bsub[local_row][local_col] = B[(bk * BLOCK_SIZE + local_row) * N + col];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      sum += Asub[local_row][k] * Bsub[k][local_col];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  C[row * N + col] = sum;
}
