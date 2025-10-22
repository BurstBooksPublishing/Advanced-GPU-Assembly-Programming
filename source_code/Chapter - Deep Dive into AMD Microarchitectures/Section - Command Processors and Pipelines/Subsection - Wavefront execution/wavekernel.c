__kernel void wave_reduce(__global float *in, __global float *out, __local float *scratch) {
    // Assumes work-group size is 64 (wave64) for best occupancy.
    const uint lid = get_local_id(0);    // local id inside WG
    const uint gid = get_global_id(0);   // global linear id

    // Load element (coalesced per wave): each wave performs contiguous loads.
    float val = in[gid];                 // single read per work-item

    // Branchless mask: replace conditional with arithmetic to avoid divergence.
    // Example: clamp negative values to zero without if-branches.
    val = val * (val > 0.0f);            // compiler emits select, not branch.

    // Intra-wave reduction using scratch (LDS) and barrier.
    // Each work-item writes its value to scratch; waves use contiguous scratch region.
    scratch[lid] = val;                  // coalesced LDS write
    barrier(CLK_LOCAL_MEM_FENCE);        // synchronize wave and other lanes in WG

    // Simple tree-reduction within WG; step sizes are powers-of-two.
    for (uint s = get_local_size(0) >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            // No divergence within a wave if WG size equals wave size.
            scratch[lid] += scratch[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First lane writes result.
    if (lid == 0) out[get_group_id(0)] = scratch[0];
}