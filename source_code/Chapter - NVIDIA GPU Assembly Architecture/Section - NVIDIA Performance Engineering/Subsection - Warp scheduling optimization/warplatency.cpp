#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__global__ void load_bound_kernel(const float* __restrict__ src,
                                  float* __restrict__ dst,
                                  int N, int stride) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= N) return;
  int idx = gid * stride;
  float acc = 0.0f;
  // chain of dependent loads to expose memory latency per iteration
  for (int i = 0; i < 16; ++i)
    acc = src[(idx + i) % N] + acc;  // serialized accumulation within warp
  dst[gid] = acc;
}

int main() {
  const int N = 1 << 20;
  float *d_src, *d_dst;
  cudaMalloc(&d_src, N * sizeof(float));
  cudaMalloc(&d_dst, N * sizeof(float));

  int stride = 1;  // adjust to test cache reuse or stride effects
  int blocks = (N + 255) / 256;

  std::vector<int> block_sizes = {32, 64, 128, 256};
  for (int bs : block_sizes) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    load_bound_kernel<<<blocks, bs>>>(d_src, d_dst, N, stride);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("blockSize=%d  time=%.3f ms\n", bs, ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  cudaFree(d_src);
  cudaFree(d_dst);
  return 0;
}