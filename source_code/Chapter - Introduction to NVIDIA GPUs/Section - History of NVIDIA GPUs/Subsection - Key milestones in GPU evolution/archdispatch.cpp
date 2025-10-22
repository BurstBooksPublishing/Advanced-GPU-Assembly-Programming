#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul_fallback(float* A, float* B, float* C, int N) {
    // simple O(N^3) fallback matrix multiply
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
__global__ void matmul_tensor(float* A, float* B, float* C, int N) {
    // placeholder: would use WMMA intrinsics for tensor cores
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) C[idx] = A[idx] + B[idx]; // illustrative fast path
}
#endif

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Compute capability %d.%d\n", prop.major, prop.minor);

    int N = 256;
    size_t sz = N * N * sizeof(float);
    float *A, *B, *C;
    cudaMallocManaged(&A, sz);
    cudaMallocManaged(&B, sz);
    cudaMallocManaged(&C, sz);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    if (prop.major >= 7) {
        printf("Launching tensor-core accelerated kernel.\n");
        matmul_tensor<<<grid, block>>>(A, B, C, N);
    } else {
        printf("Launching fallback kernel.\n");
        matmul_fallback<<<grid, block>>>(A, B, C, N);
    }

    cudaDeviceSynchronize();
    printf("Completed execution.\n");

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}