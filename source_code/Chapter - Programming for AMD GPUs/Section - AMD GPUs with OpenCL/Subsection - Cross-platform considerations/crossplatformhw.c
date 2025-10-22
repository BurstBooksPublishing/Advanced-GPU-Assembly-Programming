/* Host-side OpenCL setup (concise, error checks omitted for clarity) */
#include 
const char *kernel_src =
"__kernel void sum_reduce(__global const float *in, __global float *out, __local float *sbuf) {\n"
"  int gid = get_global_id(0);\n"
"  int lid = get_local_id(0);\n" 
"  int lsize = get_local_size(0);\n"
"  sbuf[lid] = in[gid]; // store to LDS\n" 
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  for (int stride = lsize/2; stride > 0; stride >>= 1) {\n"
"    if (lid < stride) sbuf[lid] += sbuf[lid + stride];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"  }\n"
"  if (lid == 0) out[get_group_id(0)] = sbuf[0];\n"
"}\n";

void launch_kernel_cl(cl_device_id dev, cl_context ctx, cl_command_queue q,
                      cl_mem in_buf, cl_mem out_buf, size_t N) {
  char vendor[256]; clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
  size_t max_wg; clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
  // Choose a local size that favors AMD wave64 but remains multiple of 32 for NVIDIA.
  size_t local = (strstr(vendor, "Advanced Micro Devices") ? 64 : 32);
  if (local > max_wg) local = max_wg; // fallback
  // Create program, kernel, set args, allocate local mem and enqueue NDRange.
  cl_program prog = clCreateProgramWithSource(ctx, 1, &kernel_src, NULL, NULL);
  clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
  cl_kernel k = clCreateKernel(prog, "sum_reduce", NULL);
  clSetKernelArg(k, 0, sizeof(cl_mem), &in_buf);
  clSetKernelArg(k, 1, sizeof(cl_mem), &out_buf);
  clSetKernelArg(k, 2, local * sizeof(float), NULL); // local buffer
  size_t global = ((N + local - 1) / local) * local;
  clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL);
  // Cleanup omitted.
}