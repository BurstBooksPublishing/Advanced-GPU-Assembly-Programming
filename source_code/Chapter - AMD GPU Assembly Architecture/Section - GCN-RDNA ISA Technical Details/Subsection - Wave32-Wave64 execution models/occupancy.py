# occupancy.py -- compute residency limits for wave32 and wave64
def occupancy(R_V, R_S, LDS, W_hw, r_v, r_s, r_lds):
    # returns (W32, W64) resident waves per CU
    def W_for(L):
        w_v = R_V // (L * r_v)         # VGPR-limited waves
        w_s = R_S // r_s               # SGPR-limited waves (per-wave)
        w_l = LDS // r_lds             # LDS-limited waves (per-wave)
        return min(w_v, w_s, w_l, W_hw)
    return W_for(32), W_for(64)

# Example: hypothetical CU resources (fill with targets for your GPU)
R_V = 25600     # total VGPRs per CU (example value)
R_S = 512       # total SGPRs per CU (example value)
LDS = 65536     # bytes of LDS per CU
W_hw = 40       # hardware-imposed max resident waves per CU

# Sweep per-thread VGPR usage (r_v) and show change in occupancy
for r_v in [2,4,8,16,32]:
    r_s = 8       # SGPRs per wave (choose according to compiler output)
    r_lds = 1024  # bytes of LDS per wave
    w32, w64 = occupancy(R_V, R_S, LDS, W_hw, r_v, r_s, r_lds)
    print(f"r_v={r_v:2d}: W32={w32:2d}, W64={w64:2d}")
# // Use real hardware counters / compiler reports to set R_V, R_S, etc.