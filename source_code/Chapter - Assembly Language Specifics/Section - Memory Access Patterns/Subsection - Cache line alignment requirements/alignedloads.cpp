#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

// Host: allocate 128-byte aligned source buffer (pinned could be used for efficiency).
void* aligned_alloc_128(size_t bytes) {
    void* p = nullptr;
    posix_memalign(&p, 128, bytes);
    return p;
}

// Device kernel: each warp cooperatively reads 128B segments as 8 x uint4 (16B each).
__global__ void warp_segment_load(const uint32_t* src, uint32_t* dst, size_t elems) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & 31;              // lane index in warp
  const int warp_id = tid >> 5;
  const size_t elems_per_segment = 128 / sizeof(uint32_t); // number of 32-bit elems in 128B
  const size_t seg_idx = (tid / elems_per_segment);        // segment index for this element
  const uint32_t* seg_base = src + seg_idx * elems_per_segment;

  // Within each segment, map lanes to 8 vector-load slots (uint4 == 16B)
  const int slot = lane & 7;                      // 8 slots per 128B segment
  const uint4* vbase = reinterpret_cast<const uint4*>(seg_base);
  uint4 v = vbase[slot];                          // vectorized load (requires base alignment)

  // Scatter loaded vector to per-thread locations (example reduces memory ops)
  // Each lane extracts its 32-bit element from v (illustrative mapping)
  uint32_t out = reinterpret_cast<uint32_t*>(&v)[(lane >> 3)]; // lane->element mapping
  if (tid < elems) dst[tid] = out;
}