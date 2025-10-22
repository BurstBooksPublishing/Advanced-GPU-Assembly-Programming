# Compute active waves and occupancy fraction for per-SM/CU resources.
def occupancy(R_total, regs_per_thread, threads_per_wave, W_max, threads_max):
    # waves limited by registers
    w_by_regs = R_total // (regs_per_thread * threads_per_wave)
    active_waves = min(w_by_regs, W_max, threads_max // threads_per_wave)
    return active_waves, (active_waves * threads_per_wave) / threads_max

# Example: NVIDIA Ampere-like SM (toy numbers)
R_nvidia = 65536         # total 32-bit registers per SM (approx)
threads_max_n = 2048
W_max_n = 64             # upper bound on waves per SM (toy)
print(occupancy(R_nvidia, 64, 32, W_max_n, threads_max_n))  # regs/thread=64

# Example: AMD CU (toy numbers)
R_amd = 65536            # VGPRs per CU (toy)
threads_max_a = 2048
W_max_a = 32             # waves per CU (toy)
print(occupancy(R_amd, 48, 64, W_max_a, threads_max_a))     # regs/thread=48