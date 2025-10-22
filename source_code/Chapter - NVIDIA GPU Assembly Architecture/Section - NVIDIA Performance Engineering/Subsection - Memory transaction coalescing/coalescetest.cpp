#include 
#include 

__global__ void warp_segment_count(int stride_elems, int *out) {
    const int lane = threadIdx.x & 31;           // lane in warp
    const int warp0 = (threadIdx.x >> 5) == 0;   // single warp check
    if (!warp0) return;

    const size_t S = 128;                        // segment size (bytes)
    const size_t e = sizeof(float);              // element size
    extern __shared__ unsigned long segs[];     // small bitmap per warp
    if (lane < 64) segs[lane] = 0;               // clear bitmap (64x64-bit)
    __syncwarp();

    // compute address for demonstration (base = 0 for simplicity)
    size_t addr = (size_t)(lane) * (size_t)stride_elems * e;

    // compute segment id
    unsigned long seg = (unsigned long)(addr / S);

    // set a bit in the bitmap using atomic OR on 64-bit words
    unsigned idx = seg >> 6;                     // 64-bit word index
    unsigned bit = seg & 63;
    atomicOr(&segs[idx], 1UL << bit);
    __syncwarp();

    // barrier then thread 0 counts unique bits
    if (lane == 0) {
        int count = 0;
        // assume segments touched fall into first few words for demo
        for (int i = 0; i < 4; ++i) count += __popcll(segs[i]);
        out[0] = count;                            // write result
    }
}

int main() {
    int *d_out; cudaMalloc(&d_out, sizeof(int));
    // launch one block of 32 threads, 32*8 bytes shared (~256B)
    warp_segment_count<<<1,32, 8*8*sizeof(unsigned long)>>>(1, d_out); // contiguous
    int result; cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("contiguous stride -> segments = %d\n", result);

    warp_segment_count<<<1,32, 8*8*sizeof(unsigned long)>>>(2, d_out); // stride=2
    cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("stride=2 -> segments = %d\n", result);

    cudaFree(d_out);
    return 0;
}