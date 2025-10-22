/* kernel.cl: stride probe kernel (OpenCL C) */
__kernel void stride_probe(__global uint *buf, __global ulong *out, uint stride, uint iters) {
  size_t gid = get_global_id(0);
  size_t n = get_global_size(0);
  // each work-item walks a pointer-chase style or strided array
  size_t idx = (gid * stride) % n;
  uint val = 0;
  for(uint i = 0; i < iters; i++) {
    val += buf[idx];
    idx = (idx + stride) % n;
  }
  out[gid] = val;
}

/* host.c: minimal host-side OpenCL launcher */
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) if((x) != CL_SUCCESS){fprintf(stderr,"OpenCL error at %s:%d\n",__FILE__,__LINE__);exit(1);}

int main() {
  const size_t N = 1<<20;
  const unsigned iters = 256;
  unsigned *h_buf = (unsigned*)malloc(N * sizeof(unsigned));
  for(size_t i=0;i<N;i++) h_buf[i] = i;
  // Platform and device setup
  cl_platform_id platform; cl_device_id device; cl_context ctx; cl_command_queue q;
  cl_int err;
  CHECK(clGetPlatformIDs(1, &platform, NULL));
  CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
  ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CHECK(err);
  q = clCreateCommandQueue(ctx, device, 0, &err); CHECK(err);

  // Build kernel
  FILE *f = fopen("kernel.cl","rb"); fseek(f,0,SEEK_END); size_t src_size=ftell(f); rewind(f);
  char *src=(char*)malloc(src_size+1); fread(src,1,src_size,f); fclose(f); src[src_size]='\0';
  cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &src_size, &err); CHECK(err);
  CHECK(clBuildProgram(prog, 1, &device, NULL, NULL, NULL));
  cl_kernel krn = clCreateKernel(prog, "stride_probe", &err); CHECK(err);

  // Buffers
  cl_mem d_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, N*sizeof(unsigned), h_buf, &err); CHECK(err);
  cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N*sizeof(cl_ulong), NULL, &err); CHECK(err);

  // Launch for various strides
  size_t local=64, global=N;
  for(unsigned stride=1; stride<=1024; stride*=2) {
    CHECK(clSetKernelArg(krn, 0, sizeof(cl_mem), &d_buf));
    CHECK(clSetKernelArg(krn, 1, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(krn, 2, sizeof(unsigned), &stride));
    CHECK(clSetKernelArg(krn, 3, sizeof(unsigned), &iters));
    clock_t t0 = clock();
    CHECK(clEnqueueNDRangeKernel(q, krn, 1, NULL, &global, &local, 0, NULL, NULL));
    CHECK(clFinish(q));
    clock_t t1 = clock();
    printf("stride=%u elapsed=%.3f ms\n", stride, 1000.0*(t1-t0)/CLOCKS_PER_SEC);
  }

  // Cleanup
  clReleaseMemObject(d_buf);
  clReleaseMemObject(d_out);
  clReleaseKernel(krn);
  clReleaseProgram(prog);
  clReleaseCommandQueue(q);
  clReleaseContext(ctx);
  free(h_buf); free(src);
  return 0;
}