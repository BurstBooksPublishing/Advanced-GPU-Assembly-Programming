extern "C" __global__
void tiled_compute(const float * __restrict__ in, float * __restrict__ out,
                   int N, int tile_elems) {
  extern __shared__ float sdata[];              // dynamic shared mem
  int tid = threadIdx.x;
  int lane = tid & 31;
  int blockOffset = blockIdx.x * blockDim.x * tile_elems;
  int tidGlobal = blockOffset + tid;

  int tilesPerBlock = tile_elems / blockDim.x;  // assume divisible

  // Two buffers: 0 and 1
  float *buf0 = sdata;
  float *buf1 = sdata + blockDim.x;

  // Prime: load first tile into buf0 (vectorized by 4 floats per thread where possible)
  for (int i = 0; i < tilesPerBlock; ++i) {
    int gidx = tidGlobal + i * blockDim.x;
    // Coalesced load: read contiguous floats
    buf0[tid + i * blockDim.x] = (gidx < N) ? in[gidx] : 0.0f;
  }
  __syncthreads();

  int cur = 0;
  for (int t = 0; t < tilesPerBlock; ++t) {
    // start loading next tile into the other buffer while computing
    int nextTile = t + 1;
    if (nextTile < tilesPerBlock) {
      int nextGidx = tidGlobal + nextTile * blockDim.x;
      float *target = (cur == 0) ? buf1 : buf0;
      target[tid] = (nextGidx < N) ? in[nextGidx] : 0.0f; // async style overlap
    }

    // Compute on current buffer (simple example: scale)
    float v = ((cur == 0) ? buf0[tid] : buf1[tid]);
    // Independent arithmetic chain to overlap with next load
    float res = v * 2.0f + 1.0f;
    out[tidGlobal + t * blockDim.x] = res;

    // ensure next tile is ready before swapping
    __syncthreads();
    cur ^= 1; // ping-pong
  }
}