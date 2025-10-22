extern "C" __global__ void conv3x3_shared(
  const float* __restrict__ input,    // NCHW, N==1
  const float* __restrict__ weights,  // F x C x 3 x 3
  float* __restrict__ output,         // F x H_o x W_o
  int C, int H, int W, int F)         // input channels, HxW, filters
{
  // Tile for output spatial region computed per block
  constexpr int T_H = 8; constexpr int T_W = 8; // output tile
  const int out_u0 = blockIdx.y * T_H;
  const int out_v0 = blockIdx.x * T_W;
  const int f = blockIdx.z * blockDim.z + threadIdx.z; // output channel per thread-z

  // Shared memory holds input patch: (T_H+2) x (T_W+2) x C
  extern __shared__ float smem[]; // size = (T_H+2)*(T_W+2)*C
  const int sm_stride = (T_W+2);

  // Each thread in x/y loads several elements along channels to fill shared memory.
  // Use threadIdx.x/y to cover spatial tile; threadIdx.z iterates filters.
  const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  // Load per-channel loop: iterate over channels, each load places C elements into shared mem.
  for (int c = 0; c < C; ++c) {
    // Compute global coordinates for load; use halo offset -1..+T+1
    for (int i = ty; i < (T_H+2); i += blockDim.y) {
      for (int j = tx; j < (T_W+2); j += blockDim.x) {
        int in_u = out_u0 + i - 1;
        int in_v = out_v0 + j - 1;
        float val = 0.0f;
        if (in_u >= 0 && in_u < H && in_v >= 0 && in_v < W) {
          val = input[(c*H + in_u)*W + in_v]; // N==1
        }
        int idx = (c*(T_H+2) + i)*(T_W+2) + j;
        smem[idx] = val;
      }
    }
  }
  __syncthreads();

  // Each thread computes one output pixel for its assigned filter index f
  for (int i = ty; i < T_H; i += blockDim.y) {
    for (int j = tx; j < T_W; j += blockDim.x) {
      int out_u = out_u0 + i;
      int out_v = out_v0 + j;
      if (out_u >= H || out_v >= W || f >= F) continue;
      float sum = 0.0f;
      for (int c = 0; c < C; ++c) {
        // unrolled 3x3 kernel
        int base = (c*(T_H+2) + i)*(T_W+2) + j;
        const float* wbase = weights + ((f*C + c)*3)*3;
        sum += wbase[0]*smem[base]     + wbase[1]*smem[base+1]   + wbase[2]*smem[base+2];
        sum += wbase[3]*smem[base+sm_stride] + wbase[4]*smem[base+sm_stride+1] + wbase[5]*smem[base+sm_stride+2];
        sum += wbase[6]*smem[base+2*sm_stride] + wbase[7]*smem[base+2*sm_stride+1] + wbase[8]*smem[base+2*sm_stride+2];
      }
      output[(f*H + out_u)*W + out_v] = sum;
    }
  }
}