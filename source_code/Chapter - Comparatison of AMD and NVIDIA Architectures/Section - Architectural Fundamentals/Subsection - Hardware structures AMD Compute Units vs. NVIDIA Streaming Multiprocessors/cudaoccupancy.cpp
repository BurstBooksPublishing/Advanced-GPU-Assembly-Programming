#include 
#include 

__global__ __launch_bounds__(256,4) // max 256 threads per block, min 4 blocks per SM (hint)
void vecAdd(const float *a, const float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) c[idx] = a[idx] + b[idx]; // simple compute, low register pressure
}

int main() {
  int n = 1<<20;
  // allocate device memory omitted for brevity...
  int device; cudaGetDevice(&device);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);

  int maxBlocksPerSM;
  // compute maximum active blocks per SM for this kernel given block size 256
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &maxBlocksPerSM, vecAdd, 256, 0);
  printf("Device SMs: %d, Max active blocks/SM for vecAdd: %d\n",
         prop.multiProcessorCount, maxBlocksPerSM);
  // launch kernel omitted...
  return 0;
}