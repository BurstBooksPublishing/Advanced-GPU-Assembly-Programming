#version 450
layout(local_size_x = 32, local_size_y = 1) in; // 32 threads per workgroup (one warp/wave)
struct Light { vec4 pos; vec4 color; float radius; float padding; }; // 32-byte aligned

layout(std430, binding = 0) buffer LightBuffer {
    uint lightCount;                 // total lights
    Light lights[];                  // runtime-sized array
};

layout(std430, binding = 1) buffer TileLightLists {
    // Each tile has a write pointer + list of indices; sized by host.
    uint tileOffsets[];              // preallocated offsets per tile
};

layout(push_constant) uniform PC { ivec2 screenDim; ivec2 tileDim; uint maxLightsPerTile; } pc;

// per-workgroup shared staging to collect indices (reduce SSBO writes)
shared uint s_lightCount;            // number of lights in this tile
shared uint s_indices[128];          // tuned MAX; ensure <= workgroup shared memory budget

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    uint localId = gl_LocalInvocationID.x;

    // Only one thread initializes the per-tile counter
    if (localId == 0u) {
        s_lightCount = 0u;
    }
    // ensure initialization visible to all threads
    memoryBarrierShared();
    barrier();

    // Compute tile bounds in view or screen space (host provided or computed here)
    ivec2 tileIdx = ivec2(gl_WorkGroupID.xy);
    // Iterate lights in chunks to limit register pressure; each thread tests multiple lights.
    for (uint li = localId; li < lightCount; li += gl_WorkGroupSize.x) {
        Light L = lights[li]; // single load per light per workgroup
        // cheap sphere vs tile AABB test in 2D screen space (host can provide light screen pos)
        // assume L.pos.xy = screen-space center, L.pos.z = depth; radius in L.radius
        bool intersects = /* cheap test */ (abs(L.pos.x - float(tileIdx.x*pc.tileDim.x + pc.tileDim.x*0.5)) <= L.radius)
                         && (abs(L.pos.y - float(tileIdx.y*pc.tileDim.y + pc.tileDim.y*0.5)) <= L.radius);
        if (intersects) {
            uint idx = atomicAdd(s_lightCount, 1u); // safe within workgroup
            if (idx < pc.maxLightsPerTile) {
                s_indices[idx] = li; // store global index
            }
        }
    }
    // ensure all writes to shared completed
    memoryBarrierShared();
    barrier();

    // One thread writes the compacted list to global SSBO at tile offset
    if (localId == 0u) {
        uint offset = tileIdx.x + tileIdx.y * (pc.screenDim.x / pc.tileDim.x); // compute tile id
        uint dst = tileOffsets[offset]; // preassigned base for each tile
        uint count = min(s_lightCount, pc.maxLightsPerTile);
        for (uint i = 0u; i < count; ++i) {
            // write to global list: host must layout destination buffer accordingly
            // here we assume subsequent shader will read indices from destination region
            // atomic or ordered writes may be required depending on buffer layout
            // (host-managed offsets recommended)
        }
    }
}