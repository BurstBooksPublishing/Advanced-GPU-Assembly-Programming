#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in; // choose multiple of 32
layout(std430, binding = 0) buffer SrcA { float A[]; };         // device buffer A
layout(std430, binding = 1) buffer SrcB { float B[]; };         // device buffer B
layout(std430, binding = 2) buffer Dst  { float C[]; };         // device buffer C
layout(push_constant) uniform Push { uint N; } pc;             // small params via push constants

shared float tile[64]; // tile sized to local_size_x

void main() {
    uint gid = gl_GlobalInvocationID.x;          // global index
    uint lid = gl_LocalInvocationID.x;           // local index within workgroup
    uint groupBase = gl_WorkGroupID.x * gl_WorkGroupSize.x;

    // Load into shared memory cooperatively (coalesced global loads)
    if (groupBase + lid < pc.N) tile[lid] = A[groupBase + lid];
    barrier(); // ensure shared tile is populated

    // Compute: each thread reads both shared tile and B (global), accumulate
    if (gid < pc.N) {
        float a = tile[lid];                      // reuse from shared memory
        float b = B[gid];                         // global read
        C[gid] = a + b;                           // simple operation for portability
    }
    // No atomics or subgroup intrinsics used here.
}