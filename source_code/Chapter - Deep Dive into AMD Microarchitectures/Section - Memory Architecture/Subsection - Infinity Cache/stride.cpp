#include <hip/hip_runtime.h>
#include <cstdio>

const size_t N = 1 << 26; // 64M elements (~256MB for float)

__global__ void load_kernel(float *A, size_t stride, size_t iters) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idx = (i * stride) & (N - 1);
  float s = 0.0f;
  for (size_t it = 0; it < iters; ++it) {
    s += A[(idx + it * stride) & (N - 1)];
  }
  if (i == 0) A[0] = s; // prevent optimization
}

int main() {
  float *dA;
  hipMalloc(&dA, N * sizeof(float));
  hipMemset(dA, 0, N * sizeof(float));

  const size_t threads = 256;
  const size_t blocks = 256;
  const size_t iters = 1024;

  hipEvent_t s, e;
  hipEventCreate(&s);
  hipEventCreate(&e);

  // Small stride -> high locality
  hipEventRecord(s, 0);
  load_kernel<<<blocks, threads>>>(dA, 1, iters);
  hipEventRecord(e, 0);
  hipEventSynchronize(e);
  float ms;
  hipEventElapsedTime(&ms, s, e);
  printf("stride=1 time(ms)=%.3f\n", ms);

  // Large stride -> low locality
  hipEventRecord(s, 0);
  load_kernel<<<blocks, threads>>>(dA, 8193, iters);
  hipEventRecord(e, 0);
  hipEventSynchronize(e);
  hipEventElapsedTime(&ms, s, e);
  printf("stride=8193 time(ms)=%.3f\n", ms);

  hipEventDestroy(s);
  hipEventDestroy(e);
  hipFree(dA);
  return 0;
}