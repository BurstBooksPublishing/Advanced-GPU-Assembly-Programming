#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void tiled_gemm(
    const uint M, const uint N, const uint K,
    __global const float *A, __global const float *B, __global float *C,
    const uint lda, const uint ldb, const uint ldc)
{
    const uint wg_x = get_group_id(0);            // tile column
    const uint wg_y = get_group_id(1);            // tile row
    const uint lx = get_local_id(0);              // 0..63 expected (wave64)
    const uint ly = get_local_id(1);
    const uint local_idx = ly * get_local_size(0) + lx;

    // Tile sizes tuned for Vega's LDS and wave64 grouping
    const uint TILE_M = 64;   // rows per work-group
    const uint TILE_N = 64;   // cols per work-group
    const uint TILE_K = 16;   // inner dimension tile

    // Shared local memory (LDS) for A and B tiles
    __local float Asub[TILE_M * TILE_K];
    __local float Bsub[TILE_K * TILE_N];

    // Compute global starting indices for this work-group
    const uint row = wg_y * TILE_M + ly;
    const uint col = wg_x * TILE_N + lx;

    float acc = 0.0f;

    // Loop over K in tiles
    for (uint k0 = 0; k0 < K; k0 += TILE_K) {
        // Each work-item cooperatively loads parts of A and B into LDS
        // Vectorize loads with float4 where possible for coalescing (driver will pack).
        uint a_index = row * lda + (k0 + lx);
        uint b_index = (k0 + ly) * ldb + col;

        // Guard loads for tail tiles
        if ((row < M) && (k0 + lx < K)) {
            Asub[local_idx] = A[a_index];
        } else {
            Asub[local_idx] = 0.0f;
        }
        if ((k0 + ly < K) && (col < N)) {
            Bsub[local_idx] = B[b_index];
        } else {
            Bsub[local_idx] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE); // ensure tile is resident in LDS

        // Compute inner-product over TILE_K for this tile
        for (uint kk = 0; kk < TILE_K; ++kk) {
            float aval = Asub[ly * TILE_K + kk];
            float bval = Bsub[kk * TILE_N + lx];
            acc += aval * bval;
        }
        barrier(CLK_LOCAL_MEM_FENCE); // sync before overwriting LDS
    }

    // Write result back
    if (row < M && col < N) {
        C[row * ldc + col] = acc;
    }
}