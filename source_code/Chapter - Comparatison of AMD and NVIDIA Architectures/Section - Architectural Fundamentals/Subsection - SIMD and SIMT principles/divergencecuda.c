#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__global__ void divergent_kernel(float *a, int n, float thr) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  float v = a[gid];
  // Divergent branch: half the warp may take extra work
  if (v > thr) {
    // expensive per-thread work on some lanes
    for (int i = 0; i < 64; i++) v = sinf(v) * cosf(v) + v;
  } else {
    v = v * 0.5f;
  }
  a[gid] = v;
}

__global__ void branchless_kernel(float *a, int n, float thr) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  float v = a[gid];
  // Compute predicate mask (0.0f or 1.0f) without branching
  float p = (v > thr) ? 1.0f : 0.0f;
  float extra = v;
  // perform expensive work on all lanes but hide result via mask
  for (int i = 0; i < 64; i++) extra = sinf(extra) * cosf(extra) + extra;
  // merge results: if p==1 use 'extra', else use scaled v
  a[gid] = p * extra + (1.0f - p) * (v * 0.5f);
}

int main() {
  const int N = 1 << 20;
  float *h = (float*)malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) h[i] = (float)i / N;
  float *d;
  cudaMalloc(&d, N * sizeof(float));
  cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice);

  divergent_kernel<<<(N + 255) / 256, 256>>>(d, N, 1.0f);
  cudaDeviceSynchronize();
  branchless_kernel<<<(N + 255) / 256, 256>>>(d, N, 1.0f);
  cudaDeviceSynchronize();

  cudaMemcpy(h, d, N * sizeof(float), cudaMemcpyDeviceToHost);
  printf("sample %f\n", h[1]);

  cudaFree(d);
  free(h);
  return 0;
}