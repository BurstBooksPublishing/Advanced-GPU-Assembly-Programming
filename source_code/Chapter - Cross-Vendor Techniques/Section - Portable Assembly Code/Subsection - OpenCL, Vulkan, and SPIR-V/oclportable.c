__kernel void block_copy_and_reduce(__global const float *src, __global float *dst,
                                   uint n) {
  // Work-item and group indices.
  const uint gid = get_global_id(0);
  const uint lid = get_local_id(0);
  const uint lsize = get_local_size(0);

  // Cooperative block load size: choose vector width for memory coalescing.
  typedef float4 v4;                 // 4-wide vector for better bandwidth.
  const uint elems_per_v4 = 4u;
  const uint block_v4 = (lsize + elems_per_v4 - 1) / elems_per_v4;

  // Local stash: allocate enough elements for block loads (vectorized).
  __local v4 local_buf[/* compile-time constant or tuned size */ 64];

  // Calculate element index for this work-item's first vector lane.
  uint base_v4 = (get_group_id(0) * block_v4) * elems_per_v4;
  uint v4_index = base_v4 + lid / elems_per_v4;

  // Cooperative load with bounds check; each work-item loads its v4 lane.
  v4 val = (v4)(0.0f);
  uint idx = v4_index * elems_per_v4;
  if (idx < n) {
    // safe tail load: copy element-by-element to avoid OOB reads.
    float tmp[4];
    #pragma unroll
    for (uint i=0;i<4;i++) {
      uint pos = idx + i;
      tmp[i] = (pos < n) ? src[pos] : 0.0f;
    }
    val = v4(tmp[0], tmp[1], tmp[2], tmp[3]);
  }
  local_buf[lid] = val;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Parallel reduction inside the work-group using pairwise halving.
  // This loop makes no assumption about native wave size.
  for (uint stride = lsize/2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      v4 a = local_buf[lid];
      v4 b = local_buf[lid + stride];
      local_buf[lid] = a + b; // vector add reduces four floats at once
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write group result by lane 0.
  if (lid == 0) {
    // collapse v4 to scalar sum and write (example).
    v4 r = local_buf[0];
    float sum = r.s0 + r.s1 + r.s2 + r.s3;
    dst[get_group_id(0)] = sum;
  }
}