__kernel void grouped_write(__global int *out, __local int *lds, __global int *token) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int group = get_group_id(0);
  // 1) Producer: write into LDS (fast on-chip)
  int val = compute_partial(gid);
  lds[lid] = val;               // local store
  mem_fence(CLK_LOCAL_MEM_FENCE); // ensure LDS store visible to workgroup
  barrier(CLK_LOCAL_MEM_FENCE);   // workgroup sync (maps to s_barrier)
  // 2) Consumer: cooperative reduction in LDS (no global drain)
  int reduced = workgroup_reduce(lds, lid);
  if (lid == 0) {
    // 3) Coalesce and write reduced result once per workgroup
    out[group] = reduced;       // single global store per group (coalesced)
    mem_fence(CLK_GLOBAL_MEM_FENCE); // ensures all global stores are visible
    // 4) Release token: a single atomic signals completion to other agents
    atomic_store_explicit((volatile __global int*)token + group, 1, memory_order_release);
  }
}