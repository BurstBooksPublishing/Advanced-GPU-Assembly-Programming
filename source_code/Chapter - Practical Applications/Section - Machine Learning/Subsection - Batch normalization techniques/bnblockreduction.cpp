extern "C" __global__ void bn_compute_mean_var(
    const half* __restrict__ input, // N*C*H*W, stored NCHW
    float* __restrict__ mean,       // C
    float* __restrict__ var,        // C
    int N, int C, int H, int W,
    float eps) {
  extern __shared__ float sdata[]; // dynamic: two floats per thread warp-slot
  int c = blockIdx.x;              // one channel per block
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warpId = tid >> 5;
  int warpsPerBlock = (blockDim.x + 31) / 32;

  // each block processes channel c across all N*H*W elements
  int M = N * H * W;
  float sum = 0.0f;
  float sumsq = 0.0f;
  // stride across elements so threads in block cover the channel's elements
  for (int idx = tid; idx < M; idx += blockDim.x) {
    int n = idx / (H*W);
    int hw = idx % (H*W);
    int offset = ((n*C + c)*H + (hw / W))*W + (hw % W); // NCHW linear index
    float v = __half2float(input[offset]); // convert FP16 to FP32
    sum += v;
    sumsq += v*v;
  }

  // warp-level reduction using shuffle
  for (int off = 16; off > 0; off >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
    sumsq += __shfl_down_sync(0xFFFFFFFF, sumsq, off);
  }
  // write one value per warp to shared memory
  if (lane == 0) {
    sdata[warpId*2 + 0] = sum;
    sdata[warpId*2 + 1] = sumsq;
  }
  __syncthreads();

  // final reduction by first warp
  if (warpId == 0) {
    float s = 0.0f, q = 0.0f;
    for (int i = lane; i < warpsPerBlock; i += 32) {
      s += sdata[i*2 + 0];
      q += sdata[i*2 + 1];
    }
    // warp reduce again
    for (int off=16; off>0; off>>=1) {
      s += __shfl_down_sync(0xFFFFFFFF, s, off);
      q += __shfl_down_sync(0xFFFFFFFF, q, off);
    }
    if (lane == 0) {
      // atomic add partial sums to global accumulators
      atomicAdd(&mean[c], s);
      atomicAdd(&var[c], q);
    }
  }
}