extern "C" __global__ void __launch_bounds__(256,4)
vector_fma_tiled(const float *A, const float *B, float *C, int N) {
  // tile per block to increase reuse (I=FLOPs/byte)
  __shared__ float sA[256]; // 256 floats per block tile
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int tile_base = blockIdx.x * 256;
  // load tile (coalesced)
  if (tile_base + threadIdx.x < N) sA[threadIdx.x] = A[tile_base + threadIdx.x];
  __syncthreads();
  // compute fused multiply-add with reuse from sA
  float acc = 0.0f;
  if (gid < N) {
    float b = B[gid];                // single load per thread
    for (int i = 0; i < 256 && tile_base + i < N; ++i) {
      acc = fmaf(sA[i], b, acc);     // FMA instruction (high throughput)
    }
    C[gid] = acc;
  }
}