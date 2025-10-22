__kernel void group_reduce(__global const float *input,   // global source
                           __global float *group_sums,   // per-group result
                           __local float *local_buf)    // local (LDS) buffer
{
    size_t gid = get_global_id(0);         // global index
    size_t lid = get_local_id(0);          // local index in group
    size_t group = get_group_id(0);        // work-group id
    size_t local_size = get_local_size(0); // work-items per group

    // Load element into LDS to exploit shared-bandwidth.
    float val = input[gid];
    local_buf[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE); // ensure all writes visible

    // Binary tree reduction in LDS. local_size must be power-of-two.
    for (size_t stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_buf[lid] += local_buf[lid + stride]; // combine pair
        }
        barrier(CLK_LOCAL_MEM_FENCE); // synchronize after each step
    }

    // First lane writes the per-group sum to global memory.
    if (lid == 0) {
        group_sums[group] = local_buf[0]; // single write per group
    }
}