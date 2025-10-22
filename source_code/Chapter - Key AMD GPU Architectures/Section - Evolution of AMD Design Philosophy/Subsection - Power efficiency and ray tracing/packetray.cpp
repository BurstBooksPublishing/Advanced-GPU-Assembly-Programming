extern "C" __global__ void packetRayKernel(
    const BVHNode* bvh, const Triangle* tris, Ray* rays, Hit* hits, int numRays)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;          // global ray id
  int lane = threadIdx.x & 63;                              // wave lane for Wave64
  int waveBase = gid - lane;                                // base id for wave packet

  // Gather a 64-lane packet; inactive lanes use sentinel rays.
  Ray packet[64];
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    int id = waveBase + i;
    packet[i] = (id < numRays) ? rays[id] : Ray::sentinel(); // avoid raw underscores in text
  }

  // Issue traversal to hardware-accelerated function (driver intrinsic).
  // This call avoids explicit BVH stack manipulation in wavefront ALUs.
  hwTraversePacket(bvh, tris, packet, hits + waveBase, 64); // intrinsic; driver handles mapping

  // Post-process shading for active lanes (minimal divergence).
  if (gid < numRays && hits[gid].hit) {
    // compact shading work per-lane to reduce divergent execution.
    ShadeHit(hits[gid], gid);
  }
}