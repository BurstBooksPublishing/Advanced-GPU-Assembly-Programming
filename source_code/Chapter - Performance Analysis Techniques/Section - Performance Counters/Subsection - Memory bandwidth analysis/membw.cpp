#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// simple error check macro
#define CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); \
  } \
} while(0)

__global__ void copy_kernel(uint4* dst, const uint4* src, size_t words) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (size_t idx = i; idx < words; idx += stride) {
    // 128-bit load/store using uint4 for coalesced accesses
    uint4 v = src[idx];
    dst[idx] = v;
  }
}

int main() {
  const size_t MB = 1024 * 1024;
  const size_t bytes = 512 * MB;                // 512 MiB buffer
  const size_t words = bytes / sizeof(uint4);   // number of uint4 elements

  uint4 *d_src, *d_dst;
  CHECK(cudaMalloc(&d_src, bytes));
  CHECK(cudaMalloc(&d_dst, bytes));

  // initialize source with a kernel or cudaMemset (omitted for brevity)
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  int threads = 256;
  int blocks = 1024; // tune to saturate GPU

  CHECK(cudaEventRecord(start));
  copy_kernel<<<blocks, threads>>>(d_dst, d_src, words);
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop)); // milliseconds
  double seconds = ms * 1e-3;
  double total_bytes = double(bytes); // one-way copy
  double bwGBs = (total_bytes / seconds) / 1e9; // GB/s

  printf("Copy throughput: %.3f GB/s (elapsed %.3f ms)\n", bwGBs, ms);

  CHECK(cudaFree(d_src));
  CHECK(cudaFree(d_dst));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  return 0;
}