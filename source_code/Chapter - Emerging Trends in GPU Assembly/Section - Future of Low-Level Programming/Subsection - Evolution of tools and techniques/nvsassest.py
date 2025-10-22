#!/usr/bin/env python3
import subprocess,sys,re

# usage: python3 nv_sass_est.py mykernel.cubin 1.41e9 2.0
cubin, f_sm, ipc_nom = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])

# call nvdisasm to get SASS; requires NVIDIA toolchain on PATH
out = subprocess.check_output(['nvdisasm', cubin], text=True)

# count instructions conservatively (lines beginning with opcodes)
inst_lines = [l for l in out.splitlines() if re.match(r'\s*\b[A-Z][a-zA-Z0-9_\.]+\b', l)]
N_inst = len(inst_lines)

# simple compute time estimate (Equation 1)
T_comp = N_inst / (ipc_nom * f_sm)  # seconds

print(f"Instructions: {N_inst}")
print(f"SM clock (Hz): {f_sm:.3e}, assumed IPC: {ipc_nom:.2f}")
print(f"Estimated compute-bound time: {T_comp*1e6:.3f} us")
# further steps: correlate with Nsight metrics to refine ipc_nom; autotune kernels based on hotspots