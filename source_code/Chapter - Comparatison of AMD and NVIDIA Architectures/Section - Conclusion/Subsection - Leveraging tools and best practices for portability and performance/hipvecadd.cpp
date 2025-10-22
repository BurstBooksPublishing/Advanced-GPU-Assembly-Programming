#include <hip/hip_runtime.h>
#include <cstdio>

// Tile width parameterized for autotuning.
constexpr int TILE = 128;

// Kernel: each thread handles one element; tile size used for shared staging.
__global__ void vec_add(const float* __restrict__ a, const float* __restrict__ b,
                        float* __restrict__ c, size_t n) {
  extern __shared__ float s[];           // shared memory for coalesced staging
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    // coalesced load into shared memory (example pattern)
    s[threadIdx.x] = a[gid];
    __syncthreads();
    float va = s[threadIdx.x];           // fast shared read
    c[gid] = va + b[gid];
  }
}

int main() {
  size_t n = 1 << 24;
  size_t bytes = n * sizeof(float);
  float *d_a, *d_b, *d_c;
  hipMalloc(&d_a, bytes);
  hipMalloc(&d_b, bytes);
  hipMalloc(&d_c, bytes);
  // Populate device buffers omitted for brevity.

  // Query device properties to tune block size.
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0); // device 0
  int blockSize = 256;              // default guess

  // Ask runtime for near-optimal block size (occupancy helper).
  hipOccupancyMaxPotentialBlockSize(nullptr, nullptr,
                                    (void*)vec_add, 0, 0, &blockSize);

  int grid = (n + blockSize - 1) / blockSize;
  size_t shared_bytes = blockSize * sizeof(float);
  hipLaunchKernelGGL(vec_add, dim3(grid), dim3(blockSize), shared_bytes, 0,
                     d_a, d_b, d_c, n);
  hipDeviceSynchronize();

  // Validation and cleanup.
  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);
  return 0;
}