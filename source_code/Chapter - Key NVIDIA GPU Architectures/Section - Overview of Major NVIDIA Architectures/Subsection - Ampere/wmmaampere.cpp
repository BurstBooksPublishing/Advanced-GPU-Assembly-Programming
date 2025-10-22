#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Kernel: C = A * B, A,B in FP16, C in FP32 (accumulate)
__global__ void wmma_gemm_fp16(const half *A, const half *B, float *C,
                               int M, int N, int K) {
  // Each thread block computes a 64×64 tile of C
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  // Warp identifiers
  int warpId = threadIdx.x / 32;
  int laneId = threadIdx.x % 32;

  // Warp tile coordinates within block
  int warpM = blockRow * 64 + (warpId / 4) * WMMA_M;
  int warpN = blockCol * 64 + (warpId % 4) * WMMA_N;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> bFrag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accFrag;
  wmma::fill_fragment(accFrag, 0.0f);

  // Tile over K dimension in steps of WMMA_K
  for (int k0 = 0; k0 < K; k0 += WMMA_K) {
    const half *tileA = A + warpM * K + k0;
    const half *tileB = B + k0 * N + warpN;

    wmma::load_matrix_sync(aFrag, tileA, K);
    wmma::load_matrix_sync(bFrag, tileB, N);
    wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
  }

  // Store the resulting 16×16 tile of C
  float *tileC = C + warpM * N + warpN;
  wmma::store_matrix_sync(tileC, accFrag, N, wmma::mem_row_major);
}