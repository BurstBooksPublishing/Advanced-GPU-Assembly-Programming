extern "C" __global__ void tiledCopy(const float *src, float *dst, int N) {
  extern __shared__ float sdata[];            // shared scratch, size set by launch
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x & 31;                // warp lane (assumes 32-wide warp)
  int warp_id = threadIdx.x >> 5;
  int tile_offset = blockIdx.x * blockDim.x;  // tile base in global memory

  // Each thread loads one contiguous element per iteration for coalescing
  int idx = tile_offset + threadIdx.x;
  if (idx < N) {
    sdata[threadIdx.x + warp_id] = src[idx]; // slight padding by warp_id to reduce bank conflicts
  } else {
    sdata[threadIdx.x + warp_id] = 0.0f;
  }
  __syncthreads();

  // Compute on tile using shared memory, then write back coalesced
  float val = sdata[threadIdx.x + warp_id] * 2.0f; // example compute
  if (idx < N) dst[idx] = val;
}