#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>

// Simple SGEMM: C = alpha * A * B + beta * C (column-major)
// Error checking omitted for brevity; production code must check returns.

int main() {
  const int M = 512, N = 512, K = 512;
  const float alpha = 1.0f, beta = 0.0f;
  float *dA, *dB, *dC;
  size_t sizeA = size_t(M) * K * sizeof(float);
  size_t sizeB = size_t(K) * N * sizeof(float);

  // Allocate device buffers
  hipMalloc(&dA, sizeA);
  hipMalloc(&dB, sizeB);
  hipMalloc(&dC, size_t(M) * N * sizeof(float));

  // Initialize host data and copy (omitted here; assume device buffers filled)

  // Create rocBLAS handle
  rocblas_handle handle;
  rocblas_create_handle(&handle);

  // Call rocBLAS SGEMM (no transpose)
  rocblas_sgemm(handle,
                rocblas_operation_none, rocblas_operation_none,
                M, N, K,
                &alpha,
                dA, M,   // lda = M for column-major
                dB, K,   // ldb = K
                &beta,
                dC, M);  // ldc = M

  hipDeviceSynchronize(); // Wait for completion

  rocblas_destroy_handle(handle);
  hipFree(dA);
  hipFree(dB);
  hipFree(dC);
  return 0;
}