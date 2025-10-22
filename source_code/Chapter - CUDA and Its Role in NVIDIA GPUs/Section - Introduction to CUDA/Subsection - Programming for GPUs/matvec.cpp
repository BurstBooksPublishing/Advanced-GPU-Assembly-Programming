#include <cuda_runtime.h>
#include <cstdio>

__global__ void matvec_tiled(const float* __restrict__ A,
                             const float* __restrict__ x,
                             float* y, int N) {
  extern __shared__ float sdata[];  // shared tile for vector x
  const int tid = threadIdx.x;
  const int row = blockIdx.x * blockDim.x + tid;

  float sum = 0.0f;

  for (int tile = 0; tile < N; tile += blockDim.x) {
    int col = tile + tid;
    // cooperative load of x into shared memory
    sdata[tid] = (col < N) ? x[col] : 0.0f;
    __syncthreads();

    if (row < N) {
      #pragma unroll 4
      for (int j = 0; j < blockDim.x && (tile + j) < N; ++j)
        sum += A[row * N + (tile + j)] * sdata[j];
    }
    __syncthreads();
  }

  if (row < N)
    y[row] = sum;
}

int main() {
  const int N = 4096;
  const int block = 256;
  const size_t bytes = N * N * sizeof(float);
  const size_t vecBytes = N * sizeof(float);

  float *A, *x, *y;
  cudaMallocManaged(&A, bytes);
  cudaMallocManaged(&x, vecBytes);
  cudaMallocManaged(&y, vecBytes);

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    for (int j = 0; j < N; j++)
      A[i * N + j] = (i == j) ? 1.0f : 0.0f;
  }

  dim3 grid((N + block - 1) / block);
  size_t sharedMem = block * sizeof(float);
  matvec_tiled<<<grid, block, sharedMem>>>(A, x, y, N);
  cudaDeviceSynchronize();

  printf("y[0] = %f, y[N-1] = %f\n", y[0], y[N - 1]);

  cudaFree(A);
  cudaFree(x);
  cudaFree(y);
  return 0;
}