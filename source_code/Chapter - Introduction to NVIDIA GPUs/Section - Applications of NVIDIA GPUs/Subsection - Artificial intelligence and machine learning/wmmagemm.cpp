#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

//
// Each thread block computes one 16×16 output tile of C using WMMA fragments.
// This version assumes dimensions are multiples of 16 for simplicity.
//
extern "C" __global__
void wmma_gemm_kernel(const half *A, const half *B, float *C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
  int block_tile_m = blockIdx.y;
  int block_tile_n = blockIdx.x;

  // WMMA fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  // Loop over K dimension in steps of 16
  for (int k0 = 0; k0 < K; k0 += 16) {
    const half *A_tile = A + (block_tile_m * 16) * lda + k0;
    const half *B_tile = B + k0 * ldb + (block_tile_n * 16);

    wmma::load_matrix_sync(a_frag, A_tile, lda);
    wmma::load_matrix_sync(b_frag, B_tile, ldb);

    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // Store result to global memory
  float *C_tile = C + (block_tile_m * 16) * ldc + (block_tile_n * 16);
  wmma::store_matrix_sync(C_tile, acc_frag, ldc, wmma::mem_row_major);
}