#include <hip/hip_runtime.h>
#include <cstdio>

// A and B are row-major, C = alpha*A*B + beta*C
__global__ void tiled_gemm(const double* __restrict__ A,
                           const double* __restrict__ B,
                           double* C, int N, double alpha, double beta) {
  constexpr int T = 16; // tile size
  __shared__ double sA[T][T];
  __shared__ double sB[T][T];

  int gx = blockIdx.x * T;
  int gy = blockIdx.y * T;
  int lx = threadIdx.x;
  int ly = threadIdx.y;

  double acc = 0.0;
  for (int k = 0; k < N; k += T) {
    // Load tile from A and B into LDS with coalesced accesses
    int arow = gy + ly, acol = k + lx;
    int brow = k + ly, bcol = gx + lx;
    sA[ly][lx] = (arow < N && acol < N) ? A[arow * N + acol] : 0.0;
    sB[ly][lx] = (brow < N && bcol < N) ? B[brow * N + bcol] : 0.0;
    __syncthreads();

    // Compute inner product for the tile
    #pragma unroll
    for (int t = 0; t < T; ++t)
      acc += sA[ly][t] * sB[t][lx];
    __syncthreads();
  }

  int crow = gy + ly, ccol = gx + lx;
  if (crow < N && ccol < N)
    C[crow * N + ccol] = alpha * acc + beta * C[crow * N + ccol];
}

// Host wrapper (error handling omitted for brevity)
void launch_gemm(const double* A, const double* B, double* C, int N) {
  dim3 block(16, 16); // align to wave size multiples when possible
  dim3 grid((N + 15) / 16, (N + 15) / 16);
  hipLaunchKernelGGL(tiled_gemm, grid, block, 0, 0, A, B, C, N, 1.0, 0.0);
}