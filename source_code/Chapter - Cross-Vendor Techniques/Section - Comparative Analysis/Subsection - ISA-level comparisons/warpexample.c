extern "C" __global__ void warp_reduce_sum(const float *a, float *out, int n) {
    unsigned lane = threadIdx.x & 31;               // lane within warp
    unsigned wid  = threadIdx.x >> 5;               // warp id within block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? a[idx] : 0.0f;          // guarded load

    // warp-level inclusive reduction using shuffle down; uses PTX shfl.* under the hood
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFFu, val, offset);
        val += other;                                // per-lane sum accumulation
    }

    // write one value per warp: lane 0 stores the reduced sum
    unsigned active_mask = __ballot_sync(0xFFFFFFFFu, idx < n); // maps to vote instruction
    if (lane == 0) out[blockIdx.x * (blockDim.x/32) + wid] = val;
}
// Comments: __shfl_down_sync and __ballot_sync compile to warp shuffle and vote PTX ops.