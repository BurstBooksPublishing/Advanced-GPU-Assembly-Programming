#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>

// Device reduction using AMD wave intrinsics; uses wave32-friendly patterns.
__global__ void wave_reduce(const float* __restrict__ in, float* out, int N) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  float val = (gid < N) ? in[gid] : 0.0f;

  // Get lane id and wavefront size via AMD builtins (works on ROCm/GFX targets).
  unsigned lane = __builtin_amdgcn_wavelaneid();       // lane within wave
  unsigned waveSz = __builtin_amdgcn_wavefrontsize();  // expected 32 on RDNA

  // Perform tree reduction within the wave using lane reads (compiler maps to ds intrinsics).
  for (unsigned offset = waveSz >> 1; offset > 0; offset >>= 1) {
    float other = __builtin_amdgcn_readlane(val, (lane + offset) % waveSz);
    val += other;
  }

  // Lane 0 of each wave writes the partial sum to out indexed by wave ID.
  if (lane == 0) {
    unsigned waveId = (blockIdx.x * blockDim.x + threadIdx.x) / waveSz;
    out[waveId] = val; // partial sums later reduced by host or second kernel
  }
}

// Host driver (minimal): allocates data, launches kernel, reads back partials.
int main() {
  const int N = 1 << 20;
  float *h_in = (float*)malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) h_in[i] = 1.0f;

  float *d_in, *d_out;
  hipMalloc(&d_in, N * sizeof(float));
  hipMemcpy(d_in, h_in, N * sizeof(float), hipMemcpyHostToDevice);

  int numWaves = N / 32;
  hipMalloc(&d_out, numWaves * sizeof(float));

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  hipLaunchKernelGGL(wave_reduce, grid, block, 0, 0, d_in, d_out, N);
  hipDeviceSynchronize();

  float *h_out = (float*)malloc(numWaves * sizeof(float));
  hipMemcpy(h_out, d_out, numWaves * sizeof(float), hipMemcpyDeviceToHost);

  // Verify
  float total = 0.0f;
  for (int i = 0; i < numWaves; i++) total += h_out[i];
  printf("Reduction result: %f (expected %d)\n", total, N);

  hipFree(d_in);
  hipFree(d_out);
  free(h_in);
  free(h_out);
  return 0;
}