#include 

__global__ void axpy_shared_uniform(float a, const float* x, float* y, int n) {
    extern __shared__ float s[];           // shared buffer for the uniform 'a'
    if (threadIdx.x == 0) s[0] = a;        // single lane stores uniform -> reduces per-thread reg use
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float mul = s[0];                  // read uniform from shared (1 load instead of per-thread reg)
        y[i] = mul * x[i] + y[i];          // compute; compiler may use fewer registers per-thread
    }
}
// launch: axpy_shared_uniform<<>>(a, x, y, n);