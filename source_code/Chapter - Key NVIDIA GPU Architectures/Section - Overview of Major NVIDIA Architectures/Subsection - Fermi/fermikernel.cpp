#include <cuda_runtime.h>
#include <stdio.h>

// Limit registers and threads per block for predictable occupancy.
// (maxThreadsPerBlock = 256, minBlocksPerSM = 4)
__launch_bounds__(256, 4)
__global__ void tiledVecAdd(const float *A, const float *B, float *C, size_t N) {
    extern __shared__ float sdata[];   // dynamic shared memory tile
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;

    // Coalesced load from global memory
    if (gid < N)
        sdata[tid] = A[gid] + B[gid];
    else
        sdata[tid] = 0.0f;
    __syncthreads();

    // Shared-memory reduction (stride-halving pattern)
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write one result per block
    if (tid == 0)
        C[blockIdx.x] = sdata[0];
}

int main() {
    const size_t N = 1 << 20;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    const size_t shmSize = blockSize * sizeof(float);

    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(gridSize * sizeof(float));

    for (size_t i = 0; i < N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, gridSize * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    tiledVecAdd<<<gridSize, blockSize, shmSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Block 0 result: %f\n", h_C[0]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}