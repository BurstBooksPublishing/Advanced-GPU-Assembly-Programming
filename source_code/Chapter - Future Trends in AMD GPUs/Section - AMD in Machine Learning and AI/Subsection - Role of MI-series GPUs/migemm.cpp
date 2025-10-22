#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <cstdio>

int main() {
  const int N = 1024;                // square matrices for simplicity
  const float alpha = 1.0f, beta = 0.0f;
  std::vector<float> A(N * N), B(N * N), C(N * N);
  // initialize host matrices (omitted for brevity)...

  float *dA, *dB, *dC;
  hipMalloc(&dA, N * N * sizeof(float)); // device allocation
  hipMalloc(&dB, N * N * sizeof(float));
  hipMalloc(&dC, N * N * sizeof(float));

  hipMemcpy(dA, A.data(), N * N * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(dB, B.data(), N * N * sizeof(float), hipMemcpyHostToDevice);

  rocblas_handle handle;
  rocblas_create_handle(&handle);    // create rocBLAS handle

  // perform C = alpha * A * B + beta * C
  rocblas_sgemm(handle,
                rocblas_operation_none, rocblas_operation_none,
                N, N, N,
                &alpha,
                dA, N,
                dB, N,
                &beta,
                dC, N);

  hipMemcpy(C.data(), dC, N * N * sizeof(float), hipMemcpyDeviceToHost);
  // validate or use C...
  rocblas_destroy_handle(handle);
  hipFree(dA);
  hipFree(dB);
  hipFree(dC);
  return 0;
}