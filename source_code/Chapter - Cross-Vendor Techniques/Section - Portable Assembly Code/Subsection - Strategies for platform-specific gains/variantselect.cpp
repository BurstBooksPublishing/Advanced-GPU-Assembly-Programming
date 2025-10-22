#include <cstdio>
#ifdef __CUDACC__
  #include <cuda_runtime.h>
#else
  #include <hip/hip_runtime.h>
  #define cudaGetDeviceProperties hipGetDeviceProperties  // HIP compatibility macro
  #define cudaSetDevice hipSetDevice
#endif

// Forward-declared tuned kernel variants.
__global__ void kernel_variantA(float* __restrict__ out, const float* __restrict__ in, int N); // tuned for small R_t
__global__ void kernel_variantB(float* __restrict__ out, const float* __restrict__ in, int N); // tuned for large tile

int main() {
  int dev = 0;
  cudaSetDevice(dev);

#ifdef __CUDACC__
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);            // NVIDIA device properties
#else
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, dev);             // AMD/ROCm device properties
#endif

  int warpSize     = prop.warpSize;               // wavefront/warp size
  int regsPerSM    = prop.regsPerMultiprocessor;
  int sharedPerSM  = prop.sharedMemPerMultiprocessor;

  printf("Device: %s\n", prop.name);
  printf("Warp/Wave Size: %d, Regs/SM: %d, SharedMem/SM: %d KB\n",
         warpSize, regsPerSM, sharedPerSM / 1024);

  // Simple cost model for two variants
  int threadsPerBlockA = 128, threadsPerBlockB = 256;
  int regsPerThreadA = 32, regsPerThreadB = 24; // measured offline

  // Proxy for occupancy (resident warps per SM)
  int residentA = regsPerSM / (regsPerThreadA * threadsPerBlockA / warpSize);
  int residentB = regsPerSM / (regsPerThreadB * threadsPerBlockB / warpSize);

  // Select variant maximizing resident warps
  bool chooseA = (residentA >= residentB);
  printf("Variant A estimated warps/SM: %d\n", residentA);
  printf("Variant B estimated warps/SM: %d\n", residentB);
  printf("Chosen variant: %s\n", chooseA ? "A" : "B");

  // Example launch (dummy data, demonstration only)
  if (chooseA)
    kernel_variantA<<<1024 / threadsPerBlockA, threadsPerBlockA>>>(nullptr, nullptr, 0);
  else
    kernel_variantB<<<1024 / threadsPerBlockB, threadsPerBlockB>>>(nullptr, nullptr, 0);

  return 0;
}