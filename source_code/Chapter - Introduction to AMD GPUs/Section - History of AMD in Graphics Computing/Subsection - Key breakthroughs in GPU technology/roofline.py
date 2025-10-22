# Compute break-even intensity and throughput for multiple kernels.
# Units: FLOPS in TFLOP, bandwidth in GB/s, intensities in FLOP/byte.
def roofline_peak_throughput(P_TF, B_GB, intensities):
    P = P_TF * 1e12            # FLOP/s
    B = B_GB * 1e9             # byte/s
    results = {}
    for I in intensities:
        T = min(P, B * I)      # achievable FLOP/s (eq. 1)
        results[I] = {'throughput_TF': T/1e12, 'bound': 'compute' if T==P else 'memory'}
    return results

# Example: GPU with 16 TFLOPS peak. Compare 512 GB/s (HBM) vs 1024 GB/s (HBM+InfinityCache effect).
intensities = [1, 8, 32, 128]  # FLOP/byte
print(roofline_peak_throughput(16, 512, intensities))   # HBM baseline
print(roofline_peak_throughput(16, 1024, intensities))  # effective larger BW