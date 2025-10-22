#!/usr/bin/env python3
import pandas as pd
import sys

# Usage: python ncu_analyze.py ncu_output.csv sm_clock_hz num_sms max_warps
csv = sys.argv[1]
sm_clock_hz = float(sys.argv[2])      # SM clock in Hz
num_sms = int(sys.argv[3])           # number of SMs
max_warps = int(sys.argv[4])         # max warps per SM

df = pd.read_csv(csv)                 # NCU CSV with metric columns

# Replace these strings if your tool uses different names
warp_pct_col = 'sm__warps_active.avg.pct'        # percent of warps active
cycles_col = 'sm__cycles_elapsed.avg'           # cycles elapsed (avg)
dram_read_col = 'dram__bytes_read.sum'          # DRAM bytes read
dram_write_col = 'dram__bytes_write.sum'        # DRAM bytes written

# Compute occupancy (fraction)
occupancy_frac = df[warp_pct_col].mean() / 100.0

# Convert cycles to seconds using SM clock
cycles = df[cycles_col].sum()
time_seconds = cycles / sm_clock_hz

# Sum DRAM bytes and compute effective bandwidth (GB/s)
bytes_read = df.get(dram_read_col, pd.Series([0])).sum()
bytes_write = df.get(dram_write_col, pd.Series([0])).sum()
bandwidth_gb = (bytes_read + bytes_write) / time_seconds / 1e9

print(f"achieved_occupancy={occupancy_frac:.3f}")
print(f"elapsed_time(s)={time_seconds:.6f}")
print(f"effective_bandwidth(GB/s)={bandwidth_gb:.3f}")