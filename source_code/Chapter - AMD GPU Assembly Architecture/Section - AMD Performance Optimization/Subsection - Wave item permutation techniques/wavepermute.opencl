#pragma OPENCL EXTENSION cl_khr_subgroups : enable

__kernel void wave_permute(__global const uint *in,
                           __global uint *out,
                           const uint N) {
  size_t gid = get_global_id(0);
  // lane and subgroup helpers
  uint lane = get_sub_group_local_id();           // lane within subgroup
  uint sgsize = get_sub_group_size();            // W
  // load lane-local value (one element per lane)
  uint val = (gid < N) ? in[gid] : 0u;

  // Butterfly permutation: iterative XOR-based exchanges
  for (uint offset = 1u; offset < sgsize; offset <<= 1u) {
    // partner lane computed by XOR; sub_group_shuffle_xor provides exchange
    uint partner_val = sub_group_shuffle_xor(val, offset);
    // Merge rule: example here swaps based on bit test (illustrative)
    // Keep lower index value; application-specific merge may differ.
    if ((lane & offset) != 0u) {
      val = partner_val;
    }
    // implicit barrier inside subgroup for shuffle semantics
  }

  // Example of a final arbitrary index remap per lane (index_map provided externally)
  // index_map is an sg-size mapping broadcast to each subgroup via global memory
  // For demonstration we compute a rotation: new_index = (lane + 3) % sgsize
  uint new_index = (lane + 3u) % sgsize;
  uint final_val = sub_group_shuffle(val, new_index); // single-stage fetch

  // write result back to global memory
  if (gid < N) out[gid] = final_val;
}