__attribute__((reqd_work_group_size(64,1,1))) // enforce work-group == wavefront
__kernel void wf_reduce(__global const float* A, __global float* out, __local float* sdata) {
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);    // lane id within wavefront (0..63)
    // load and perform per-lane work (vector ALU friendly)
    float val = A[gid];                  // coalesced global load
    // intra-wave pairwise reduction using LDS and barrier
    sdata[lid] = val;                    // write to LDS (L1-backed on GCN)
    barrier(CLK_LOCAL_MEM_FENCE);        // ensure LDS write visible to wave
    // tree reduce in LDS (power-of-two wavefront)
    for (uint offset = 32; offset > 0; offset >>= 1) {
        if (lid < offset) {
            sdata[lid] += sdata[lid + offset]; // reuse VGPRs, simple scalar ops
        }
        barrier(CLK_LOCAL_MEM_FENCE);    // synchronize within wave
    }
    if (lid == 0) out[get_group_id(0)] = sdata[0]; // one write per work-group
}