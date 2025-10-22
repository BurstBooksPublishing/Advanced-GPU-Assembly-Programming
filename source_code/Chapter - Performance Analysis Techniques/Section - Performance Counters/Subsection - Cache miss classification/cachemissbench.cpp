#include 
#include 
__global__ void stride_kernel(int32_t *a, int n, int stride, int iters) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int N = n;
  // Each thread walks its own strided sequence to reduce cross-thread interference.
  for (int it = 0; it < iters; ++it) {
    int idx = (tid * stride + it) & (N - 1); // N must be power-of-two for mask.
    // simple read-modify-write to exercise load/store paths.
    int v = a[idx];
    v += 1;
    a[idx] = v;
  }
}
int main() {
  const int threads = 256, blocks = 64;
  for (int exp = 16; exp <= 26; ++exp) { // sweep working set 64KB..64MB
    int N = 1 << exp;
    int size = N * sizeof(int32_t);
    int32_t *d_a;
    cudaMalloc(&d_a, size); cudaMemset(d_a, 0, size);
    for (int stride = 1; stride <= 1024; stride <<= 1) {
      // Launch kernel; collect profiler counters externally per-launch.
      stride_kernel<<>>(d_a, N, stride, 256);
      cudaDeviceSynchronize();
      // small sleep or marker could be used to separate profiling samples.
    }
    cudaFree(d_a);
  }
  return 0;
}