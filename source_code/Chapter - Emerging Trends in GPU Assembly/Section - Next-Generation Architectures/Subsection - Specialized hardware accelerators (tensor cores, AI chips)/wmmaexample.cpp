#include 
#include 
using namespace nvcuda::wmma;

// Simple 16x16x16 tile GEMM using FP16 inputs and FP32 accumulation.
__global__ void wmma_gemm(const half *A, const half *B, float *C,
                          int M, int N, int K) {
  // Block coordinates tile the output matrix by 16x16.
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;
  int row = block_row * 16;
  int col = block_col * 16;

  // Fragments: A is row-major, B is col-major for efficient access.
  fragment a_frag;
  fragment b_frag;
  fragment c_frag;
  fill_fragment(c_frag, 0.0f); // initialize accumulator fragment

  // Iterate over K in 16-wide tiles.
  for (int k0 = 0; k0 < K; k0 += 16) {
    // Load A and B sub-tiles into WMMA fragments (handles alignment).
    load_matrix_sync(a_frag, A + row * K + k0, K);
    load_matrix_sync(b_frag, B + k0 * N + col, N);

    // Single instruction at hardware level: fused matrix multiply-accumulate.
    mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  // Store the accumulator back to global memory in row-major order.
  store_matrix_sync(C + row * N + col, c_frag, N, mem_row_major);
}