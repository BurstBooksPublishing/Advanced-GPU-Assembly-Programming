extern "C" __global__
void vec_add_unroll4(const float* __restrict__ a,
                     const float* __restrict__ b,
                     float* __restrict__ c,
                     unsigned int n) {
    // thread-global index and per-thread stride
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // pointer start for this thread
    const float* pa = a + gid;
    const float* pb = b + gid;
    float* pc = c + gid;

    // main loop: process 4 elements per iteration
    unsigned int i = gid;
    for (; i + 4*stride - 1 < n; i += 4*stride) {
        // load 4 elements (coalesced per thread group)
        float a0 = *pa; float b0 = *pb;
        float a1 = *(pa + stride); float b1 = *(pb + stride);
        float a2 = *(pa + 2*stride); float b2 = *(pb + 2*stride);
        float a3 = *(pa + 3*stride); float b3 = *(pb + 3*stride);

        // compute - note these are independent (ILP)
        float r0 = a0 + b0;
        float r1 = a1 + b1;
        float r2 = a2 + b2;
        float r3 = a3 + b3;

        // store results
        *pc = r0;
        *(pc + stride) = r1;
        *(pc + 2*stride) = r2;
        *(pc + 3*stride) = r3;

        // advance pointers
        pa += 4*stride; pb += 4*stride; pc += 4*stride;
    }

    // tail loop: remaining elements
    for (; i < n; i += stride) {
        *pc = *pa + *pb;
        pa += stride; pb += stride; pc += stride;
    }
}