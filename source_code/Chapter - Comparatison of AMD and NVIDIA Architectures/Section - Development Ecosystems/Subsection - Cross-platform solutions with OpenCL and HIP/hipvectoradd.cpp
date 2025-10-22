#include 
#include 
#include 

// Simple vector add kernel
__global__ void vecAdd(const float* A, const float* B, float* C, size_t N) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N) C[gid] = A[gid] + B[gid];
}

int main() {
  const size_t N = 1 << 20;
  size_t bytes = N * sizeof(float);
  float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes), *hC = (float*)malloc(bytes);
  for (size_t i = 0; i < N; ++i) { hA[i] = (float)i; hB[i] = 1.0f; }

  float *dA, *dB, *dC;
  hipMalloc(&dA, bytes); hipMalloc(&dB, bytes); hipMalloc(&dC, bytes);
  hipMemcpy(dA, hA, bytes, hipMemcpyHostToDevice); hipMemcpy(dB, hB, bytes, hipMemcpyHostToDevice);

  // Choose block size aligned to platform scheduling quantum
  #if defined(__HIP_PLATFORM_NVCC__)
    const int quantum = 32; // NVIDIA warp
  #elif defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
    const int quantum = 64; // AMD GCN wavefront (RDNA may be 32; tune if needed)
  #else
    const int quantum = 32; // safe default
  #endif

  // target block size: 256 threads but aligned to quantum
  int block = ((256 + quantum - 1) / quantum) * quantum;
  int grid = (N + block - 1) / block;

  // Launch kernel
  hipLaunchKernelGGL(vecAdd, dim3(grid), dim3(block), 0, 0, dA, dB, dC, N);
  hipDeviceSynchronize();

  hipMemcpy(hC, dC, bytes, hipMemcpyDeviceToHost);
  // Verify a few elements
  for (int i = 0; i < 5; ++i) printf("C[%d]=%f\n", i, hC[i]);

  hipFree(dA); hipFree(dB); hipFree(dC);
  free(hA); free(hB); free(hC);
  return 0;
}