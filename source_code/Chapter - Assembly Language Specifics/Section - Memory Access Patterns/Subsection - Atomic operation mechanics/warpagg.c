extern "C" __global__
void warp_aggregate_atomic(unsigned long long *global_sum, const unsigned int *vals, size_t N) {
  // Each thread loads an element (if in bounds), does a warp shuffle reduction,
  // one thread per warp writes a partial sum to shared memory, then one warp
  // per block reduces those and performs one atomicAdd to global_sum.
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long thread_val = 0ULL;
  if (idx < N) thread_val = (unsigned long long)vals[idx]; // load

  unsigned mask = 0xFFFFFFFFu; // full-warp mask
  // Warp-level shuffle reduction (CUDA 9+): reduce within 32-thread warp
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    unsigned long long other = __shfl_down_sync(mask, thread_val, offset);
    thread_val += other; // RMW within register space
  }

  // lane 0 of each warp writes its warp-sum into shared memory
  __shared__ unsigned long long warp_sums[32]; // supports up to 1024 threads/block
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  if (lane == 0) warp_sums[warp_id] = thread_val;
  __syncthreads(); // make warp_sums visible to the block

  // Let warp 0 of the block reduce the per-warp sums
  unsigned long long block_partial = 0ULL;
  if (warp_id == 0) {
    // ThreadIdx.x < number of warps in block; check bounds
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < num_warps) block_partial = warp_sums[threadIdx.x];
    // reduce within the first warp
    for (int offset = warpSize/2; offset > 0; offset /= 2)
      block_partial += __shfl_down_sync(mask, block_partial, offset);
    // lane 0 of warp 0 issues the global atomic
    if (threadIdx.x == 0) {
      // one atomicAdd per block rather than per thread
      atomicAdd((unsigned long long*)global_sum, block_partial); // global atomic
    }
  }
  // end kernel
}