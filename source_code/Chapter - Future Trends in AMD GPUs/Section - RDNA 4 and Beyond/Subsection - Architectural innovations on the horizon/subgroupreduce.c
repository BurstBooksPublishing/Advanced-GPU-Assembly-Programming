#pragma OPENCL EXTENSION cl_khr_subgroups : enable

__kernel void subgroup_reduce_sum(__global const float* in, __global float* out, const uint N) {
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    // subgroup id and size (maps to wavefront on AMD)
    const uint sg_id = get_sub_group_id();
    const uint sg_sz = get_sub_group_size();

    // Load element; if out-of-bounds, treat as zero.
    float val = (gid < N) ? in[gid] : 0.0f;

    // First perform fast subgroup reduction (hardware accelerated on RDNA-like GPUs).
    float sg_sum = sub_group_reduce_add(val);

    // One lane per subgroup writes to local memory for cross-subgroup reduction.
    if (get_sub_group_local_id() == 0) {
        // local_buf indexed by subgroup id within workgroup
        __local float local_buf[64]; // enough for many subgroups
        local_buf[sg_id] = sg_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        // single work-item completes final reduction (min overhead).
        if (lid == 0) {
            float s = 0.0f;
            const uint nsg = (get_local_size(0) + sg_sz - 1) / sg_sz;
            for (uint i = 0; i < nsg; ++i) s += local_buf[i];
            out[get_group_id(0)] = s; // per-workgroup partial sum
        }
    }
}