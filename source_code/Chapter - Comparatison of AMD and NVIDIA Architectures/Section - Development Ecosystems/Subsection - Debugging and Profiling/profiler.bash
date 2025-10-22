# NVIDIA: system-level timeline (low overhead) and compute metrics for hot kernel
nsys profile --trace=cuda,nvtx --output=trace_gpu ./app    # timeline with NVTX ranges
ncu --export=app.ncu-rep --target-processes all ./app      # deep compute metrics capture

# Disassemble cubin to view SASS (NVIDIA)
cuobjdump -sass ./kernel.cubin > kernel.sass               # inspect instruction bundles

# AMD: timeline and counters using ROCm tools
# Capture timeline and HW counters with rocprofiler (or RGP GUI for detailed traces)
rocprof --stats ./app                                      # lightweight stats capture
rocprof --hsa-trace ./app > trace_amd                       # HSA timeline trace

# Disassemble AMD code object (ROCm)
rocdisasm ./kernel.co > kernel.sasm                         # view ISA-level code