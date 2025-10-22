#include <cuda_runtime.h>
#include <cstdio>

// Vector add with warp-level reduction demo.
// Each thread computes c[i] = a[i] + b[i]; then warp reduces sum of results.
__global__ void vecAddWarpReduce(const float *a, const float *b, float *c,
                                 float *warpSums, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;

    if (gid < n)
        val = a[gid] + b[gid];

    if (gid < n)
        c[gid] = val;

    // Warp-level reduction using __shfl_down_sync (full warp mask)
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    // Lane 0 writes warp sum
    int lane = threadIdx.x & 31;
    int warpId = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x >> 5);
    if (lane == 0)
        warpSums[warpId] = val;
}

int main() {
    const int N = 1 << 20;
    const int TPB = 256;
    size_t bytes = N * sizeof(float);

    float *a, *b, *c, *warpSums;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    int warpsPerBlock = TPB / 32;
    int numWarps = ((N + TPB - 1) / TPB) * warpsPerBlock;
    cudaMallocManaged(&warpSums, numWarps * sizeof(float));

    // Occupancy query: how many blocks per SM can be active
    int maxBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocks, vecAddWarpReduce, TPB, 0);
    printf("Max active blocks/SM for TPB=%d : %d\n", TPB, maxBlocks);

    // Launch kernel
    int blocks = (N + TPB - 1) / TPB;
    vecAddWarpReduce<<<blocks, TPB>>>(a, b, c, warpSums, N);
    cudaDeviceSynchronize();

    printf("Result example: c[0]=%f warpSum[0]=%f\n", c[0], warpSums[0]);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(warpSums);
    return 0;
}