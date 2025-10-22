#include <cuda_runtime.h>
#include <cstdio>

// Simple element-wise kernel (vector add) — complete and compilable.
__global__ void vecAdd(const float* a, const float* b, float* c, size_t n) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x; // global thread ID
    if (gid < n)
        c[gid] = a[gid] + b[gid];
}

int main() {
    size_t n = 1 << 24;                    // ~16 million elements
    int blockSize = 256;                   // threads per block
    int gridSize = (n + blockSize - 1) / blockSize;

    // Query maximum active blocks per SM
    int activeBlocksPerSM = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM, vecAdd, blockSize, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Occupancy query failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int smCount = prop.multiProcessorCount;
    int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / 32;

    // Calculate occupancy
    int warpsPerBlock = (blockSize + 31) / 32;
    int activeWarpsPerSM = activeBlocksPerSM * warpsPerBlock;
    float occupancy = float(activeWarpsPerSM) / float(maxWarpsPerSM);

    printf("Device: %s\n", prop.name);
    printf("SMs=%d, Active blocks/SM=%d, Active warps/SM=%d, Occupancy=%.2f\n",
           smCount, activeBlocksPerSM, activeWarpsPerSM, occupancy);

    // Allocate memory and run kernel
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}