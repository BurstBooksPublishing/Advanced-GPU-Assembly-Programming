#include <hip/hip_runtime.h>
#include <cstdio>

const int N = 1 << 20;

__global__ void vec_add(float* a, float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  float *A, *B, *C;
  hipMalloc(&A, N * sizeof(float));
  hipMalloc(&B, N * sizeof(float));
  hipMalloc(&C, N * sizeof(float));

  // number of concurrent streams
  const int nStreams = 4;
  hipStream_t s[nStreams];
  for (int i = 0; i < nStreams; ++i) hipStreamCreate(&s[i]);

  // chunk size per stream
  int chunk = N / nStreams;

  // Example host arrays (static initialization for brevity)
  float *hA = new float[N];
  float *hB = new float[N];
  for (int i = 0; i < N; ++i) {
    hA[i] = static_cast<float>(i);
    hB[i] = static_cast<float>(2 * i);
  }

  // Launch async copies and kernels per stream
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * chunk;
    // async host-to-device copies
    hipMemcpyAsync(A + offset, hA + offset, chunk * sizeof(float),
                   hipMemcpyHostToDevice, s[i]);
    hipMemcpyAsync(B + offset, hB + offset, chunk * sizeof(float),
                   hipMemcpyHostToDevice, s[i]);
    // launch kernel on each stream (queue-level concurrency)
    dim3 g((chunk + 255) / 256);
    dim3 b(256);
    hipLaunchKernelGGL(vec_add, g, b, 0, s[i], A + offset, B + offset, C + offset, chunk);
    // async device-to-host copy
    hipMemcpyAsync(hA + offset, C + offset, chunk * sizeof(float),
                   hipMemcpyDeviceToHost, s[i]);
  }

  // synchronize all streams
  for (int i = 0; i < nStreams; ++i) hipStreamSynchronize(s[i]);

  // cleanup
  for (int i = 0; i < nStreams; ++i) hipStreamDestroy(s[i]);
  hipFree(A);
  hipFree(B);
  hipFree(C);
  delete[] hA;
  delete[] hB;

  printf("Completed multi-stream HIP execution.\n");
  return 0;
}