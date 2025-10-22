__kernel void vec_add(__global const int *A, __global const int *B,
                      __global int *C, const int N) {
  size_t gid = get_global_id(0);        // typically compiled to VGPR (per-lane)
  if (gid < (size_t)N) {                // bound-check uses scalar compare + branch
    int a = A[gid];                      // translated to flat_load -> VGPR
    int b = B[gid];                      // translated to flat_load -> VGPR
    int r = a + b;                       // v_add_i32: VALU instruction, per-lane VGPRs
    C[gid] = r;                          // flat_store -> writes back from VGPR
  }
}