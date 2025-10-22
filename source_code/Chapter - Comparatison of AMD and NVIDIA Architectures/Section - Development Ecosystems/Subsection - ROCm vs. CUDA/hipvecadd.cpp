#include    // HIP runtime header

__global__ void vecAdd(const float* A, const float* B, float* C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // thread index
    if (idx < N) C[idx] = A[idx] + B[idx];              // compute
}

int main() {
    const size_t N = 1<<20;
    const size_t bytes = N * sizeof(float);
    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes), *hC = (float*)malloc(bytes);
    // initialize inputs...
    float *dA, *dB, *dC;
    hipMalloc(&dA, bytes); hipMalloc(&dB, bytes); hipMalloc(&dC, bytes); // allocate on device
    hipMemcpy(dA, hA, bytes, hipMemcpyHostToDevice);                    // copy to device
    hipMemcpy(dB, hB, bytes, hipMemcpyHostToDevice);

    const int block = 256;
    const int grid  = (N + block - 1) / block;
    hipLaunchKernelGGL(vecAdd, dim3(grid), dim3(block), 0, 0, dA, dB, dC, N); // launch
    hipMemcpy(hC, dC, bytes, hipMemcpyDeviceToHost);                         // copy back

    // cleanup
    hipFree(dA); hipFree(dB); hipFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}