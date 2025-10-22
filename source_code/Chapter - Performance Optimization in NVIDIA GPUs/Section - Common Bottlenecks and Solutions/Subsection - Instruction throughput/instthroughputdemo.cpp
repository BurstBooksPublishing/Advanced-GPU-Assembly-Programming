extern "C" __global__ void inst_throughput_demo(const float * __restrict__ a,
                                                const float * __restrict__ b,
                                                float * __restrict__ out,
                                                int N) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= N) return;

  // Create multiple independent accumulators to expose ILP.
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  // Process four elements per loop iteration to keep FMA pipelines busy.
  for (int i = gid; i < N; i += gridDim.x * blockDim.x * 4) {
    // use __ldg for read-only cache to reduce load latency jitter.
    float va0 = __ldg(&a[i + 0]);
    float vb0 = __ldg(&b[i + 0]);
    float va1 = __ldg(&a[i + 1]);
    float vb1 = __ldg(&b[i + 1]);
    // Fused multiply-adds compress multiply+add into one instruction where supported.
    acc0 = fmaf(va0, vb0, acc0); // independent FMA on acc0
    acc1 = fmaf(va1, vb1, acc1); // independent FMA on acc1

    float va2 = __ldg(&a[i + 2]);
    float vb2 = __ldg(&b[i + 2]);
    float va3 = __ldg(&a[i + 3]);
    float vb3 = __ldg(&b[i + 3]);
    acc2 = fmaf(va2, vb2, acc2); // keeps multiple FMA pipelines utilized
    acc3 = fmaf(va3, vb3, acc3);
  }

  // Reduce independent accumulators into a single result.
  float sum = ((acc0 + acc1) + (acc2 + acc3));
  out[gid] = sum;
}