__kernel void mem_throughput(__global uint *dst, __global uint *src,
                             uint iterations, uint stride, uint N) {
  size_t gid = get_global_id(0);
  // simple interleaved index to exercise coalescing patterns
  size_t idx = (gid * stride) % N;
  uint acc = 0u;
  for (uint i = 0; i < iterations; ++i) {
    // read then write to avoid write-combining hiding latency
    uint v = src[idx];
    acc += v;                    // prevent optimizer removing loads
    dst[idx] = v ^ (uint)i;      // simple data-dependent store
    idx = (idx + stride) % N;    // next offset
  }
  // write back per-workitem checksum
  dst[gid] = acc;
}