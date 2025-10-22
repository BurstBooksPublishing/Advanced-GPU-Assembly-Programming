#include 

extern "C" __global__ void stage_to_lds(const float* __restrict__ g_in,
                                         float* __restrict__ g_out,
                                         int N) {
  // Workgroup (block) and thread indices
  const int tid = threadIdx.x;
  const int block_offset = blockIdx.x * blockDim.x;
  const int gid = block_offset + tid;

  // Allocate LDS (shared memory) for the tile; size multiple of warp/wave
  extern __shared__ float s_tile[]; // declare dynamic shared memory

  // Cooperative load into LDS: each thread copies a stride element.
  // Guard against out-of-range; this is a production-ready boundary check.
  int global_idx = gid;
  if (global_idx < N) {
    // One coalesced global load per thread; avoids repeated global accesses later.
    s_tile[tid] = g_in[global_idx]; // cached at load time, then used from LDS
  } else {
    s_tile[tid] = 0.0f;
  }

  __syncthreads(); // ensure all tile data available in LDS

  // Inner-loop computation uses LDS (fast, bypasses L1/L2 for intra-block reuse)
  float acc = 0.0f;
  // Example: multiple passes over the tile to justify staging cost
  for (int pass = 0; pass < 8; ++pass) {
    // Access is from s_tile (LDS), not from global caches
    acc += s_tile[(tid + pass) & (blockDim.x - 1)]; // cheap, cache-friendly
  }

  __syncthreads(); // synchronize before write-back if needed

  if (global_idx < N) {
    g_out[global_idx] = acc; // single global write-back
  }
}