__kernel void tiled_vec_add(__global const float* A, __global const float* B,
                            __global float* C, uint N) {
    const uint gid = get_global_id(0);         // global thread id
    const uint lid = get_local_id(0);          // lane within workgroup
    const uint lsize = get_local_size(0);      // workgroup size
    __local float tileA[256];                  // local memory tile (LDS/shared)
    __local float tileB[256];

    // Cooperative load into local memory (coalesced global reads)
    for (uint i = lid; i < lsize && (get_group_id(0)*lsize + i) < N; i += get_local_size(0)) {
        tileA[i] = A[get_group_id(0)*lsize + i]; // coalesced load
        tileB[i] = B[get_group_id(0)*lsize + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);              // workgroup sync

    // Compute using local memory — maps to shared/LDS ALU inputs
    if (gid < N) {
        C[gid] = tileA[lid] + tileB[lid];      // local ALU operations
    }
    // no explicit global memory fence required for single-kernel semantics
}