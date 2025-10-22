__kernel void dot_prod(__global const float* A, __global const float* B,
                       __global float* out, const uint N) {
  // local sum per workgroup in LDS (use multiples of 32 for local size).
  __local float lsum[256];              // small, power-of-two for bank friendliness

  const uint gid = get_global_id(0);    // global thread id
  const uint lid = get_local_id(0);     // local thread id (0..WG-1)
  float acc = 0.0f;

  // Stride loop: each thread processes multiple elements to amortize memory latency.
  for (uint i = gid; i < N; i += get_global_size(0)) {
    // aligned loads encourage RDNA L1 coalescing; ensure A,B allocated 16B-aligned.
    acc += A[i] * B[i];
  }

  // reduction into LDS: each thread writes its partial sum.
  lsum[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Tree reduce in local memory; assume workgroup size is power-of-two and <=256.
  for (uint offset = get_local_size(0) >> 1; offset > 0; offset >>= 1) {
    if (lid < offset) {
      lsum[lid] += lsum[lid + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // thread 0 writes the result for this workgroup.
  if (lid == 0) out[get_group_id(0)] = lsum[0];
}