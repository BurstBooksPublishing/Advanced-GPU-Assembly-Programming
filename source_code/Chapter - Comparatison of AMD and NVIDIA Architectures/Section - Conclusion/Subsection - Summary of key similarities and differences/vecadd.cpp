extern "C" __global__ void vecAdd(const float* a, const float* b, float* c, int n) {
  // compute global linear index (portable across CUDA and HIP runtimes)
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;               // bounds check for arbitrary grid size
  // single instruction compute; on PTX/SASS this typically lowers to fadd
  c[i] = a[i] + b[i];               // memory access pattern must be coalesced
}
/* Host launch example (select block size with attention to occupancy):
   dim3 block(256); dim3 grid((n + block.x - 1)/block.x);
   vecAdd<<>>(dA,dB,dC,n);
   On ROCm/HIP, the same source often compiles with hipcc or hipify utilities. */