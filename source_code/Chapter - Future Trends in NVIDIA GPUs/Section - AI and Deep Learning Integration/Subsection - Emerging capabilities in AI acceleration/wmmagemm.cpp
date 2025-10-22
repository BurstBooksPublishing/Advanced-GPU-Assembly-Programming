#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
using namespace nvcuda::wmma;

extern "C" __global__ void wmma_gemm(const half *A, const half *B, float *C,
                                     int M, int N, int K) {
  // Block tile dimensions: 128×128×16 (8×8 WMMA tiles of 16×16×16)
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 16;
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  int row = blockRow * BLOCK_M + threadIdx.y * 16;
  int col = blockCol * BLOCK_N + threadIdx.x * 16;

  // Shared memory buffers for A and B subtiles
  extern __shared__ half shmem[];
  half *shA = shmem;
  half *shB = shmem + BLOCK_M * BLOCK_K;

  // WMMA fragments (accumulator in FP32)
  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> acc_frag;
  fill_fragment(acc_frag, 0.0f);

  for (int kb = 0; kb < K; kb += BLOCK_K) {
    // Load A subtile into shared memory
    int a_row = blockRow * BLOCK_M + threadIdx.y * 16;
    int a_col = kb + threadIdx.x * 16;
    for (int i = 0; i < 16; ++i) {
      int r = a_row + i;
      int c = a_col;
      if (r < M && c < K)
        shA[(threadIdx.y * 16 + i) * BLOCK_K + threadIdx.x * 16] = A[r * K + c];
    }

    // Load B subtile into shared memory
    int b_row = kb + threadIdx.y * 16;
    int b_col = blockCol * BLOCK_N + threadIdx.x * 16;
    for (int i = 0; i < 16; ++i) {
      int r = b_row + i;
      int c = b_col;
      if (r < K && c < N)
        shB[(threadIdx.y * 16 + i) * BLOCK_N + threadIdx.x * 16] = B[r * N + c];
    }

    __syncthreads();

    // Perform WMMA multiply-accumulate
    load_matrix_sync(a_frag, shA + (threadIdx.y * 16) * BLOCK_K, BLOCK_K);
    load_matrix_sync(b_frag, shB + (threadIdx.x * 16), BLOCK_N);
    mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    __syncthreads();
  }

  // Store accumulator results back to global memory
  int c_row = row;
  int c_col = col;
  float out_frag[16 * 16];
  store_matrix_sync(out_frag, acc_frag, 16, mem_row_major);

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int r = c_row + i;
      int c = c_col + j;
      if (r < M && c < N)
        C[r * N + c] = out_frag[i * 16 + j];
    }
  }
}