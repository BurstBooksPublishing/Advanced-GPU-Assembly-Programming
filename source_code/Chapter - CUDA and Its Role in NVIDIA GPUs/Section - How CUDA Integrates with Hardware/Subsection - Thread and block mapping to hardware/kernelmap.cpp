#include <cuda_runtime.h>
#include <cstdio>

__global__ void mapKernel(int *out_tlin, int *out_warp, int *out_lane, int *out_smid) {
    // compute linear thread index in block
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int bx = blockDim.x, by = blockDim.y;
    int tlin = tz * (by * bx) + ty * bx + tx;
    out_tlin[blockIdx.x * blockDim.x * blockDim.y * blockDim.z + tlin] = tlin;

    // warp and lane (warpSize = 32)
    int warp = tlin / 32;               // warp id within block
    int lane = tlin & 31;               // lane id (bitwise)
    out_warp[blockIdx.x * bx * by * blockDim.z + tlin] = warp;
    out_lane[blockIdx.x * bx * by * blockDim.z + tlin] = lane;

    // read SM ID via PTX special register %smid
    unsigned sm;
    asm volatile("mov.u32 %0, %smid;" : "=r"(sm));
    out_smid[blockIdx.x * bx * by * blockDim.z + tlin] = (int)sm;
}