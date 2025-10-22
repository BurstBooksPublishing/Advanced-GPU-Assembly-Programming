#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel_example(float *a) { /* small compute kernel */ }

int main() {
  int device; cudaGetDevice(&device);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
  int blockSize = 256; // threads per block
  int gridSize = 1024;
  int maxActiveBlocks = 0;
  // Query maximum active blocks per SM for this kernel and block size
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks,
                 kernel_example, blockSize, 0 /* dynamic shmem */);
  int warpsPerBlock = (blockSize + 31) / 32;
  int activeWarpsPerSM = maxActiveBlocks * warpsPerBlock;
  int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / 32;
  float occupancy = 100.0f * activeWarpsPerSM / maxWarpsPerSM;
  std::cout << "Active blocks/SM: " << maxActiveBlocks << "\n";
  std::cout << "Active warps/SM: " << activeWarpsPerSM << " / " << maxWarpsPerSM
            << " (" << occupancy << "% occupancy)\n";
  return 0;
}