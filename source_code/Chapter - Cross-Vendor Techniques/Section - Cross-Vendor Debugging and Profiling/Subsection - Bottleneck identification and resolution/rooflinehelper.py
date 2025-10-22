# Simple roofline-based bottleneck detector (use profiler outputs as inputs)
def analyze_kernel(peak_gflops, peak_gb_s, flops, bytes_transferred, kernel_time_s,
                   active_warps=None, max_warps=64, active_waves=None, max_waves=40):
    # convert units to consistent scale
    P_c = peak_gflops * 1e9           # FLOPs/s
    P_b = peak_gb_s * 1e9             # bytes/s
    AI = flops / bytes_transferred    # FLOPs per byte
    est_from_bandwidth = AI * P_b
    roofline_bound = min(P_c, est_from_bandwidth)
    achieved_gflops = flops / kernel_time_s / 1e9
    bottleneck = "compute" if achieved_gflops / (peak_gflops) > 0.6 else "bandwidth" \
                 if est_from_bandwidth < P_c else "mixed"
    # occupancy estimate
    occupancy = None
    if active_warps is not None:
        occupancy = active_warps / max_warps
    if active_waves is not None:
        occupancy = active_waves / max_waves
    return {
        "AI": AI, "roofline_bound_FLOPS": roofline_bound,
        "achieved_GFLOPS": achieved_gflops, "bottleneck": bottleneck,
        "occupancy_est": occupancy
    }

# Example usage (replace numbers with profiler measurements)
# print(analyze_kernel(14000, 900, flops=1e12, bytes_transferred=2e9, kernel_time_s=0.05))