// dot-prod-test.c
#include <CL/cl.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4096 // You can increase this as needed
#define BLOCK_SIZE 16
#define KERNEL_FILE "tests/matmul.cl"

float A[N * N], B[N * N], C[N * N];

void init_matrices() {
  for (int i = 0; i < N * N; i++) {
    A[i] = (float)rand() / RAND_MAX;
    B[i] = (float)rand() / RAND_MAX;
    C[i] = 0.0f;
  }
}

char *load_kernel_source(const char *filename) {
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Failed to open kernel file: %s\n", filename);
    exit(1);
  }
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  rewind(fp);
  char *src = (char *)malloc(size + 1);
  fread(src, 1, size, fp);
  src[size] = '\0';
  fclose(fp);
  return src;
}

void run_kernel(const char *kernel_name, size_t local_size[2]) {
  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  cl_mem bufA, bufB, bufC;

  clGetPlatformIDs(1, &platform, NULL);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

  char *source = load_kernel_source(KERNEL_FILE);
  program =
      clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    char log[4096];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log),
                          log, NULL);
    printf("Build log:\n%s\n", log);
    exit(1);
  }

  kernel = clCreateKernel(program, kernel_name, &err);
  bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N * N, A, &err);
  bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N * N, B, &err);
  bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * N, NULL,
                        &err);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
  int n = N;
  clSetKernelArg(kernel, 3, sizeof(int), &n);

  size_t global_size[2] = {N, N};

  double start = omp_get_wtime();
  clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0,
                         NULL, NULL);
  clFinish(queue);
  double end = omp_get_wtime();
  printf("%s: %f sec\n", kernel_name, end - start);

  clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * N * N, C, 0,
                      NULL, NULL);

  clReleaseMemObject(bufA);
  clReleaseMemObject(bufB);
  clReleaseMemObject(bufC);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(source);
}

int main() {
  srand(time(NULL));
  init_matrices();

  printf("Running with matrix size %d x %d\n", N, N);

  size_t naive_local[2] = {16, 16};
  size_t tiled_local[2] = {BLOCK_SIZE, BLOCK_SIZE};

  run_kernel("matmul_naive", naive_local);
  run_kernel("matmul_tiled", tiled_local);

  return 0;
}
