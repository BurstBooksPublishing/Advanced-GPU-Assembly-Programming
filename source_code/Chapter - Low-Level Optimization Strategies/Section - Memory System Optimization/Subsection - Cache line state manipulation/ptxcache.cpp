extern "C" __global__ void cache_policy_kernel(const int *src, int *dst, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  int val;
  // Inline PTX load: ld.global.cg.u32 loads with cache hint for L1 (cached).
  asm volatile ("ld.global.cg.u32 %0, [%1];" : "=r"(val) : "l"(src + idx)); // read with reuse hint
  val = val * 2; // compute
  // Inline PTX store: st.global.wt.u32 is a streaming (write-through) store that avoids dirty L1.
  asm volatile ("st.global.wt.u32 [%0], %1;" :: "l"(dst + idx), "r"(val)); // streaming write
}
% // end of kernel (comments are inline)