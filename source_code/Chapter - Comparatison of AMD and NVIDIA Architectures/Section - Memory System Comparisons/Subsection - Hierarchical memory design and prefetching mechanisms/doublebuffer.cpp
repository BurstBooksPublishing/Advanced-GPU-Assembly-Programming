extern "C" __global__
void tiled_prefetch_kernel(const float * __restrict__ A, float * __restrict__ B,
                           int N, int tileWidth) {
  extern __shared__ float smem[];               // dynamically sized shared mem
  int tx = threadIdx.x, bx = blockIdx.x;
  int gid = bx * blockDim.x + tx;
  int tilesPerBlock = (N + tileWidth - 1) / tileWidth;

  // double-buffer pointers in shared memory
  float *buf0 = smem;                           // tile buffer 0
  float *buf1 = smem + tileWidth;               // tile buffer 1
  int cur = 0;

  // prefetch first tile synchronously into buf0 using read-only cache (__ldg).
  int base = bx * tileWidth;
  for (int i = tx; i < tileWidth && base + i < N; i += blockDim.x) {
    // __ldg reads go through read-only cache/TEX path on NVIDIA
    buf0[i] = __ldg(&A[base + i]);              // // prefetch into shared
  }
  __syncthreads();

  for (int t = 0; t < tilesPerBlock; ++t) {
    // compute on current tile in buf0 or buf1
    float v = 0.0f;
    int localN = min(tileWidth, N - (base + t * tileWidth));
    for (int i = tx; i < localN; i += blockDim.x) {
      // simple compute example: copy with some op
      float x = (cur == 0) ? buf0[i] : buf1[i];
      v = x * 2.0f;                              // placeholder compute
      B[base + t*tileWidth + i] = v;
    }

    // prefetch next tile into the other buffer (non-blocking with respect to compute)
    int nextBase = base + (t+1) * tileWidth;
    if (nextBase < N) {
      float *dst = (cur == 0) ? buf1 : buf0;
      for (int i = tx; i < tileWidth && nextBase + i < N; i += blockDim.x) {
        dst[i] = __ldg(&A[nextBase + i]);       // next-tile loads
      }
    }
    __syncthreads();                             // ensure next tile loaded
    cur = 1 - cur;                               // swap buffers
  }
}