#include <hip/hip_runtime.h>
#include <cstdio>

constexpr int TILE_M = 64; // rows in C tile
constexpr int TILE_N = 64; // cols in C tile
constexpr int TILE_K = 16; // depth per step

__global__ void tiled_gemm(const float* A, const float* B, float* C,
                           int M, int N, int K, int lda, int ldb, int ldc) {
  // Block indices select the output tile
  int block_m = blockIdx.y;
  int block_n = blockIdx.x;
  int thread_row = threadIdx.y; // subdivide tile among threads
  int thread_col = threadIdx.x;

  // Compute tile origin
  int row0 = block_m * TILE_M;
  int col0 = block_n * TILE_N;

  // Per-thread accumulators for a small sub-tile (e.g., 4x4)
  constexpr int SUB_M = 4, SUB_N = 4;
  float acc[SUB_M][SUB_N] = {0.0f};

  // Shared memory: double buffer for A and B tiles
  __shared__ float sA[2][TILE_M * TILE_K / 16]; // sized to fit banks (simple example)
  __shared__ float sB[2][TILE_K * TILE_N / 16];

  int num_k_steps = (K + TILE_K - 1) / TILE_K;
  int buf = 0;

  // Loop over K in chunks, preload first tiles
  for (int ks = 0; ks < num_k_steps; ++ks) {
    int k0 = ks * TILE_K;

    // Each thread cooperatively loads a portion of A and B into LDS
    int a_load_row = row0 + thread_row * SUB_M;
    int a_load_col = k0 + thread_col * SUB_N;
    for (int i = 0; i < SUB_M; ++i) {
      int r = a_load_row + i;
      for (int j = 0; j < SUB_N; ++j) {
        int c = a_load_col + j;
        float val = 0.0f;
        if (r < M && c < K) val = A[r * lda + c]; // bounds-safe read
        sA[buf][(thread_row * SUB_M + i) * TILE_K + (thread_col * SUB_N + j)] = val;
      }
    }

    int b_load_row = k0 + thread_row * SUB_M;
    int b_load_col = col0 + thread_col * SUB_N;
    for (int i = 0; i < SUB_M; ++i) {
      int r = b_load_row + i;
      for (int j = 0; j < SUB_N; ++j) {
        int c = b_load_col + j;
        float val = 0.0f;
        if (r < K && c < N) val = B[r * ldb + c];
        sB[buf][(thread_row * SUB_M + i) * TILE_N + (thread_col * SUB_N + j)] = val;
      }
    }

    __syncthreads(); // ensure tiles resident

    // Compute on this K chunk
    for (int kk = 0; kk < TILE_K; ++kk) {
      for (int i = 0; i < SUB_M; ++i) {
        for (int j = 0; j < SUB_N; ++j) {
          float a_val = sA[buf][(thread_row * SUB_M + i) * TILE_K + kk];
          float b_val = sB[buf][kk * TILE_N + (thread_col * SUB_N + j)];
          acc[i][j] += a_val * b_val;
        }
      }
    }

    __syncthreads(); // synchronize before overwriting buffers
    buf ^= 1; // switch buffer for next preload (double-buffering)
  }

  // Write accumulators back to global memory
  for (int i = 0; i < SUB_M; ++i) {
    int r = row0 + thread_row * SUB_M + i;
    for (int j = 0; j < SUB_N; ++j) {
      int c = col0 + thread_col * SUB_N + j;
      if (r < M && c < N) {
        C[r * ldc + c] = acc[i][j];
      }
    }
  }
}
// Host-side launch omitted for brevity (allocate device buffers and call hipLaunchKernelGGL)