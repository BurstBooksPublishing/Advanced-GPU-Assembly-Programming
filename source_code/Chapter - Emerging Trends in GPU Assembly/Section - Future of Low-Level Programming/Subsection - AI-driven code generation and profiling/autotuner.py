import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import csv, time

def make_kernel(unroll, tile):
    # generate a simple tiled matrix multiply with parameterized unroll and tile
    src = f"""
    extern "C" __global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {{
      __shared__ float sA[{tile}][{tile}];
      __shared__ float sB[{tile}][{tile}];
      int bx = blockIdx.x, by = blockIdx.y;
      int tx = threadIdx.x, ty = threadIdx.y;
      int row = by * {tile} + ty;
      int col = bx * {tile} + tx;
      float acc = 0.0f;

      for (int m = 0; m < N / {tile}; ++m) {{
        sA[ty][tx] = A[row * N + m * {tile} + tx];
        sB[ty][tx] = B[(m * {tile} + ty) * N + col];
        __syncthreads();

        #pragma unroll {unroll}
        for (int k = 0; k < {tile}; ++k)
          acc += sA[ty][k] * sB[k][tx];

        __syncthreads();
      }}
      C[row * N + col] = acc;
    }}
    """
    return src

def autotune(N=1024, tiles=[8,16,32], unrolls=[1,2,4]):
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    C = np.zeros_like(A)
    dA = cuda.mem_alloc(A.nbytes)
    dB = cuda.mem_alloc(B.nbytes)
    dC = cuda.mem_alloc(C.nbytes)
    cuda.memcpy_htod(dA, A)
    cuda.memcpy_htod(dB, B)

    results = []
    for tile in tiles:
        for unroll in unrolls:
            src = make_kernel(unroll, tile)
            mod = SourceModule(src)
            func = mod.get_function("matmul_tiled")

            grid = (N // tile, N // tile, 1)
            block = (tile, tile, 1)
            start = cuda.Event(); end = cuda.Event()
            start.record()
            func(dA, dB, dC, np.int32(N), block=block, grid=grid)
            end.record()
            end.synchronize()
            ms = start.time_till(end)
            gflops = (2 * N**3) / (ms * 1e6)
            results.append((tile, unroll, ms, gflops))
            print(f"tile={tile}, unroll={unroll}, time={ms:.3f} ms, {gflops:.2f} GFLOP/s")

    with open("autotune_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tile", "unroll", "time_ms", "gflops"])
        writer.writerows(results)

if __name__ == "__main__":
    autotune()