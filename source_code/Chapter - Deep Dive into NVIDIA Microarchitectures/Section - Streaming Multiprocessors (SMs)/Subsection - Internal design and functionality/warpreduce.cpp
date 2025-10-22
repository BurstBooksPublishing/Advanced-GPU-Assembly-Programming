extern "C" __global__
void warp_reduce_sum(const float * __restrict__ in, float * __restrict__ out, int N) {
  extern __shared__ float sdata[];        // per-block shared memory
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x & 31;            // lane in warp (0..31)
  int warpId = threadIdx.x >> 5;         // warp index within block

  // each thread loads one element if in range
  float val = (tid < N) ? in[tid] : 0.0f;

  // warp-local reduction using shuffle (no shared mem, low latency)
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffffu, val, offset); // warp-synchronous

  // warp leader writes warp sum to shared memory
  if (lane == 0)
    sdata[warpId] = val;                  // one store per warp

  __syncthreads();                        // ensure all warp sums stored

  // first warp reduces values in shared memory
  if (warpId == 0) {
    float wsum = (threadIdx.x < (blockDim.x >> 5)) ? sdata[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      wsum += __shfl_down_sync(0xffffffffu, wsum, offset);
    if (lane == 0)
      out[blockIdx.x] = wsum;             // per-block result
  }
}