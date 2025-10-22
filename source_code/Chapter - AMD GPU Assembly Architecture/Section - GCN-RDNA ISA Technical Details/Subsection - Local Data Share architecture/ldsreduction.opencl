__kernel void wg_reduce(__global const float *g_in, __global float *g_out,
                       const unsigned int N) {
  // per-workgroup shared memory (LDS)
  __local float lbuf[256 + 32]; // pad to break simple bank aliasing

  const unsigned int gid = get_global_id(0);
  const unsigned int lid = get_local_id(0);   // lane within WG
  const unsigned int lsize = get_local_size(0); // WG size

  // each thread loads one element (guarded)
  float v = (gid < N) ? g_in[gid] : 0.0f;

  // write into LDS with padding: map index -> index + (index/BANK_STRIDE)
  unsigned int padded_idx = lid + (lid >> 5); // adds 1 per 32 lanes
  lbuf[padded_idx] = v; // single-cycle store to LDS

  barrier(CLK_LOCAL_MEM_FENCE); // ensure all writes complete

  // tree reduction inside LDS (assumes power-of-two lsize)
  for (unsigned int stride = lsize >> 1; stride > 0; stride >>= 1) {
    if (lid < stride) {
      unsigned int a = lid + (lid >> 5);
      unsigned int b = lid + stride + ((lid + stride) >> 5);
      lbuf[a] += lbuf[b]; // local read-modify-write
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    g_out[get_group_id(0)] = lbuf[0]; // write block result
  }
}