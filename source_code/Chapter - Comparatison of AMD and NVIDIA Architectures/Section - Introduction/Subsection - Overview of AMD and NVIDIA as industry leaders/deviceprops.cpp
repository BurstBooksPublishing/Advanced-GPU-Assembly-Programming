#include <cstdio>
#ifdef __HIP_PLATFORM_HCC__  // HIP/ROCm build
#include <hip/hip_runtime.h>
int main() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0); // query device 0
  std::printf("Device: %s\n", prop.name);            // device name
  std::printf("warpSize: %d\n", prop.warpSize);      // execution granularity
  std::printf("MPs: %d\n", prop.multiProcessorCount); // CUs/SMs
  std::printf("GlobalMem: %llu\n", (unsigned long long)prop.totalGlobalMem);
  return 0;
}
#else                        // CUDA build
#include <cuda_runtime.h>
int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0); // query device 0
  std::printf("Device: %s\n", prop.name);            // device name
  std::printf("warpSize: %d\n", prop.warpSize);      // execution granularity
  std::printf("MPs: %d\n", prop.multiProcessorCount); // SM count
  std::printf("GlobalMem: %llu\n", (unsigned long long)prop.totalGlobalMem);
  return 0;
}
#endif