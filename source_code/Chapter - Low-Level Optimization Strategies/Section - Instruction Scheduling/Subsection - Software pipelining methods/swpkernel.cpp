extern "C" __global__ void swp_stream(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  // Per-thread registers for pipeline stages.
  float buf_prev = 0.0f;    // previous stage result (to write)
  float buf_cur  = 0.0f;    // current compute target
  float buf_next = 0.0f;    // prefetched next input

  // Prologue: fetch first element (may be out-of-range guarded).
  int idx = tid;
  if (idx < N) buf_next = in[idx];
  idx += stride;

  // If there are additional elements, prefetch second into buf_cur.
  if (idx < N) {
    buf_cur = in[idx];
    idx += stride;
  } else {
    // Only one element: process and write then return.
    // Compute stage for single element.
    buf_cur = buf_next;
    float r = buf_cur * 2.0f;            // example compute
    out[tid] = r;                        // writeback
    return;
  }

  // Advance index for main loop prefetch.
  if (idx < N) {
    buf_prev = in[idx];                  // prefetch third stage
    idx += stride;
  } else {
    // Two elements: run simple two-iter draining loop.
    float r = buf_cur * 2.0f;            // compute cur
    out[tid] = r;
    r = buf_prev * 2.0f;                 // compute prev (which was second)
    // compute ordering preserved; write sequentially to maintain coalescing.
    out[tid + stride] = r;
    return;
  }

  // Main pipelined loop: each iteration issues prefetch, compute, write.
  while (true) {
    // Compute on buf_cur (represents iteration i).
    float result = buf_cur * 2.0f;       // example compute (ALU-bound part)

    // Prefetch next element into buf_cur for next iteration.
    if (idx < N) {
      float tmp = in[idx];               // long-latency global load
      // Rotate buffers: prev <- buf_prev (to be written), cur <- buf_next, next <- tmp
      // Here we write buf_prev (oldest) now to overlap store latency with next compute.
      out[tid + (idx - stride)] = buf_prev; // writeback overlapped (coalesced by design)
      // Rotate
      buf_prev = buf_next;
      buf_next = tmp;
      buf_cur  = buf_prev;               // prepare compute for next loop
      idx += stride;
    } else {
      // No more fetches: drain pipeline explicitly.
      out[tid + (idx - stride)] = buf_prev; // write oldest
      out[tid + (idx - 2*stride)] = result; // write current result
      break;
    }
    // The loop cycles with overlapping store, compute, and load.
    // Compiler will place these in the same basic block, enabling ILP.
  }
}