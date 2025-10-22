#include 
// coalesced load: thread i loads src[i]
__global__ void coalesced_load(const float *src, float *dst, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) dst[i] = src[i]; // contiguous per-thread access -> coalesced
}
// strided load: thread i loads src[i * stride]
__global__ void strided_load(const float *src, float *dst, int N, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * stride;
  if (idx < N) dst[i] = src[idx]; // may cause many transactions
}
/* Usage:
   - Allocate large arrays on device with cudaMalloc.
   - Launch with blockDim.x = 128 or 256 to ensure multiple warps per block.
   - Measure elapsed time with CUDA events and compare throughput.
*/