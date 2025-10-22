#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>
#include <thread>

__global__ void increment_kernel(float* data, size_t n) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) data[gid] += 1.0f; // simple per-element work
}

int main() {
  const size_t N = 1 << 20;
  float* buf = nullptr;
  hipMallocManaged(&buf, N * sizeof(float)); // unified pointer accessible to CPU/GPU

  // Initialize on CPU
  for (size_t i = 0; i < N; i++) buf[i] = static_cast<float>(i);

  // Prefetch to GPU for faster access (requires managed memory)
  int device;
  hipGetDevice(&device);
  hipMemPrefetchAsync(buf, N * sizeof(float), device, nullptr);

  // Create a HIP stream for asynchronous execution
  hipStream_t stream;
  hipStreamCreate(&stream);

  // Launch kernel asynchronously on the stream
  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  hipLaunchKernelGGL(increment_kernel, grid, block, 0, stream, buf, N);

  // Simulate CPU doing other work concurrently
  printf("CPU: Performing concurrent work while GPU computes...\n");
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Synchronize the stream (wait for kernel completion)
  hipStreamSynchronize(stream);

  // Verify result
  bool ok = true;
  for (size_t i = 0; i < N; i++) {
    if (buf[i] != static_cast<float>(i + 1)) {
      ok = false;
      printf("Mismatch at index %zu: got %f expected %f\n", i, buf[i], i + 1.0f);
      break;
    }
  }
  if (ok) printf("Verification passed: all elements incremented correctly.\n");

  // Cleanup
  hipStreamDestroy(stream);
  hipFree(buf);
  return 0;
}