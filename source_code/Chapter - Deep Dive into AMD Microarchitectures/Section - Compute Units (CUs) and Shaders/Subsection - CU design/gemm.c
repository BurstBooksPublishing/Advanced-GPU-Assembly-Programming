__attribute__((reqd_work_group_size(64,1,1)))
__kernel void tiled_gemm(const int M, const int N, const int K,
                         __global const float* A, __global const float* B,
                         __global float* C) {
  const int local_id = get_local_id(0);            // lane in a wave
  const int group_id = get_group_id(0);
  const int tile_size = 16;                        // tile per axis
  __local float sA[16*16];                         // LDS tile for A
  __local float sB[16*16];                         // LDS tile for B

  // compute global tile coords (brief comment): maps each workgroup to an output tile
  const int tile_row = group_id * tile_size;
  const int tile_col = 0;                          // single-column launch for brevity

  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += tile_size) {
    // each work-item loads one element from A and B into LDS
    int a_row = tile_row + (local_id / tile_size);
    int a_col = k0 + (local_id % tile_size);
    int b_row = k0 + (local_id / tile_size);
    int b_col = tile_col + (local_id % tile_size);

    // bounds-checked loads into local memory (LDS)
    sA[local_id] = (a_row < M && a_col < K) ? A[a_row*K + a_col] : 0.0f;
    sB[local_id] = (b_row < K && b_col < N) ? B[b_row*N + b_col] : 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);                   // ensure tile is populated

    // compute partial product for this tile (tile-size inner loop)
    for (int t = 0; t < tile_size; ++t) {
      float va = sA[(local_id / tile_size)*tile_size + t];
      float vb = sB[t*tile_size + (local_id % tile_size)];
      acc += va * vb;                               // FMA-friendly pattern
    }
    barrier(CLK_LOCAL_MEM_FENCE);                   // reuse safety
  }

  // write back result (one element per work-item)
  int out_r = tile_row + (local_id / tile_size);
  int out_c = tile_col + (local_id % tile_size);
  if (out_r < M && out_c < N) C[out_r*N + out_c] = acc;
}