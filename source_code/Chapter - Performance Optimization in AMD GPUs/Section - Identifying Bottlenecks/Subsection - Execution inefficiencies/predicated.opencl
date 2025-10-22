__kernel void reduce_predicated(__global const float* in, __global float* out,
                                __local float* local_buf, int N) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int lsize = get_local_size(0);
  float v = 0.0f;

  // Compute per-thread value without branch (predication via select)
  // Example: conditional weight, avoid 'if (cond) v += ...'
  float cond = (gid < N) ? 1.0f : 0.0f;                 // predicated mask
  float sample = cond * in[gid];                       // masked load result

  // Do more compute to increase ILP before a potential global memory op
  // chain independent arithmetic to reduce scoreboard stalls
  float t0 = sample * 0.7071f;
  float t1 = t0 * t0 + 0.125f;
  float t2 = fma(t1, 1.37f, 0.25f);                    // keep operations independent

  // Local accumulation to reduce global atomics and serialize fewer waves
  local_buf[lid] = t2;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Simple local reduction (tree) to lower number of global writes/atomics
  for (int stride = lsize/2; stride > 0; stride >>= 1) {
    if (lid < stride) local_buf[lid] += local_buf[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) out[get_group_id(0)] = local_buf[0];    // one global write per work-group
}