extern "C" __global__
void dot_kernel(const float* __restrict__ x, const float* __restrict__ y,
                float* __restrict__ out, size_t N) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int lane = threadIdx.x & 31;             // lane within warp
  unsigned int warpid = threadIdx.x >> 5;          // warp index in block
  float acc = 0.0f;

  // Each thread processes stride of grid
  size_t idx = tid;
  const size_t stride = gridDim.x * blockDim.x;
  while (idx + 3 < N) {                            // unroll 4 for throughput
    // coalesced loads: each thread touches contiguous addresses
    float a0 = x[idx];
    float b0 = y[idx];
    float a1 = x[idx+1];
    float b1 = y[idx+1];
    float a2 = x[idx+2];
    float b2 = y[idx+2];
    float a3 = x[idx+3];
    float b3 = y[idx+3];
    acc += a0*b0 + a1*b1 + a2*b2 + a3*b3;          // register accumulation
    idx += stride * 4;
  }
  // Tail cleanup
  for (; idx < N; idx += stride) acc += x[idx] * y[idx];

  // In-warp reduction using shuffle (no shared mem)
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  }

  // One thread per warp writes warp partial to shared memory
  __shared__ float warp_sums[32];                  // up to 32 warps per block
  if (lane == 0) warp_sums[warpid] = acc;
  __syncthreads();

  // First warp reduces warp_sums
  float block_sum = 0.0f;
  if (warpid == 0) {
    int num_warps = (blockDim.x + 31) / 32;
    if (threadIdx.x < num_warps) block_sum = warp_sums[threadIdx.x];
    else block_sum = 0.0f;
    // warp-level reduce again
    for (int offset = 16; offset > 0; offset >>= 1) {
      block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
    }
    if (threadIdx.x == 0) atomicAdd(out, block_sum); // one atomic per block
  }
}