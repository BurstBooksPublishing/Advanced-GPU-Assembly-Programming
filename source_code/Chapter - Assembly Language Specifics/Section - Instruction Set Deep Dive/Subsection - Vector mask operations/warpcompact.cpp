#include <cooperative_groups.h>

__device__ unsigned warp_prefix_popcount(unsigned mask, unsigned lane) {
  // returns number of 1-bits in mask for lanes < lane
  unsigned m = mask & ((1u << lane) - 1);
  return __popc(m); // fast hardware popcount
}

__global__ void compact_kernel(const int *in, int *out, int threshold, int *warp_counts) {
  // per-thread index
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x & 31;         // lane in 32-thread warp
  unsigned full_mask = 0xFFFFFFFFu;   // sync mask for whole warp

  // predicate: active if value > threshold
  bool pred = (in[gid] > threshold);

  // ballot: 32-bit mask with bits set for pred==true
  unsigned mask = __ballot_sync(full_mask, pred); // lane mask, warp-local

  // compute position within compacted segment
  unsigned pos = 0;
  if (pred) {
    pos = warp_prefix_popcount(mask, lane); // rank within warp
    // thread 0 of warp writes the warp count
    if (lane == 0) {
      warp_counts[gid / 32] = __popc(mask); // store warp active count
    }
    // compute per-warp base offset (example: prefix of warp_counts omitted for brevity)
    // here we place outputs at a per-warp contiguous region starting at warp_id*32
    int warp_base = (gid / 32) * 32;
    out[warp_base + pos] = in[gid]; // compacted write
  }
}