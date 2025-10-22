#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // compact global index
    if (idx < n) {
        // coalesced loads: adjacent threads access adjacent addresses
        float va = a[idx];
        float vb = b[idx];
        c[idx] = va + vb;  // single store, minimal register usage
    }
}

int main() {
    const int N = 1 << 20;
    size_t sz = N * sizeof(float);
    float *h_a = (float*)malloc(sz);
    float *h_b = (float*)malloc(sz);
    float *h_c = (float*)malloc(sz);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sz);
    cudaMalloc(&d_b, sz);
    cudaMalloc(&d_c, sz);

    cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, sz, cudaMemcpyDeviceToHost);

    printf("c[0] = %f, c[N-1] = %f\n", h_c[0], h_c[N-1]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}