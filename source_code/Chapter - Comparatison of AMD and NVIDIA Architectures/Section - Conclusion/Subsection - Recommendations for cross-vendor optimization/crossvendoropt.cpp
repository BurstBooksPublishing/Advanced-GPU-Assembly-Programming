#ifdef __HIPCC__
  #define WAVE 64     // AMD common default
  constexpr int TILE = 64;
#else
  #define WAVE 32     // NVIDIA warp size
  constexpr int TILE = 32;
#endif

extern "C" __global__
void tiled_gemm(const float* A, const float* B, float* C,
                int M, int N, int K, float alpha) {
  // block indices
  int bx = blockIdx.x, by = blockIdx.y;
  // thread indices
  int tx = threadIdx.x, ty = threadIdx.y;
  // tile origin in output
  int row = by * TILE + ty;
  int col = bx * TILE + tx;
  __shared__ float sA[TILE][TILE]; // shared/LDS tile
  __shared__ float sB[TILE][TILE];
  float acc = 0.0f;
  for(int k0 = 0; k0 < K; k0 += TILE) {
    // cooperative load into shared memory (coalesced by contiguous global addresses)
    int aRow = row, aCol = k0 + tx;
    int bRow = k0 + ty, bCol = col;
    // guard bounds
    sA[ty][tx] = (aRow < M && aCol < K) ? A[aRow*K + aCol] : 0.0f; // load A
    sB[ty][tx] = (bRow < K && bCol < N) ? B[bRow*N + bCol] : 0.0f; // load B
    __syncthreads(); // sync inside block/CU
    // compute inner product over tile
    for(int k = 0; k < TILE; ++k) acc += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  if(row < M && col < N) C[row*N + col] = alpha * acc;
}