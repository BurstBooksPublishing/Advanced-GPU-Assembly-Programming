extern "C" __global__ void blockAggregateAtomic(float *data, int n, float *global_sum) {
    extern __shared__ float sdata[];            // per-block shared memory
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    // load and partial sum in registers (vectorize if possible)
    float local = 0.0f;
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) local += data[i]; // coarse-grain
    sdata[tid] = local;                         // write to shared memory
    __syncthreads();                            // ensure all writes to sdata complete
    // parallel reduction (binary tree)
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) sdata[tid] += sdata[tid + offset];
        __syncthreads();                        // required to avoid race on sdata reads
    }
    if (tid == 0) {
        // Single global atomic per block reduces contention dramatically.
        atomicAdd(global_sum, sdata[0]);       // atomicAdd is coherent at device L2
        // If host or another device must immediately observe the final value,
        // call __threadfence_system() before signaling via host-visible flag.
    }
}