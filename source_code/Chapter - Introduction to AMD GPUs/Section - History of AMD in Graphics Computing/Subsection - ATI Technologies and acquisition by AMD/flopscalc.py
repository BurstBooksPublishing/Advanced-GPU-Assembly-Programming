def peak_flops(n_fma, fma_ops, clk_mhz, lanes_per_cu, num_cu):
    # n_fma: FMA units per lane; fma_ops: ops per FMA (2 for FMA)
    # clk_mhz: core clock in MHz; lanes_per_cu: lanes per compute unit
    # num_cu: number of compute units in device
    f_clk = clk_mhz * 1e6  # convert MHz to Hz
    # total FMA units across device
    total_fma = n_fma * lanes_per_cu * num_cu
    return total_fma * fma_ops * f_clk

# Example: hypothetical device parameters (values for illustration)
print(peak_flops(n_fma=1, fma_ops=2, clk_mhz=1400, lanes_per_cu=4, num_cu=40))
# => prints theoretical FLOPS (float ops per second)