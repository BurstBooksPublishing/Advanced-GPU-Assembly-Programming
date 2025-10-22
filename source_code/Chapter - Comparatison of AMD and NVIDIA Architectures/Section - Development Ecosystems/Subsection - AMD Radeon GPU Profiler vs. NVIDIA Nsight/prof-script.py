import subprocess, json, csv
# Run Nsight Compute CLI (ncu) with JSON output for kernel metrics
subprocess.run(["ncu", "--export", "ncu_report", "--target-processes", "all", "./app"], check=True)  # run app and collect
# Run RGP CLI (rgp) to capture a GPU trace; output RGPR/.json if supported
subprocess.run(["rgp", "-o", "rgp_report", "--capture", "./app"], check=True)  # capture trace

# Parse basic Nsight JSON for IPC and bandwidth (simplified)
with open("ncu_report.ncu-rep", "r") as f:
    ncu = json.load(f)  # vendor JSON format
ipc = ncu["reports"][0]["metrics"]["sm__ipc.avg"]  # sample metric key
bw_used = ncu["reports"][0]["metrics"]["dram__throughput.avg"]  # bytes/sec

# Parse RGP JSON for wave occupancy (simplified)
with open("rgp_report.json","r") as f:
    rgp = json.load(f)
w_active = rgp["summary"]["max_active_wavefronts"]  # wave count
w_max = rgp["device"]["max_wavefronts_per_cu"]

# Compute utilization components (units normalized elsewhere)
U_ipc = ipc / ncu["device"]["ipc_peak"]
U_bw  = bw_used / ncu["device"]["dram_max_bandwidth"]
U_wave = w_active / w_max
print("U_eff:", min(U_ipc, U_bw, U_wave))