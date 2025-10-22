# occupancy.py: calculate waves per CU and occupancy fraction
def compute_occupancy(vgpr_per_lane, sgpr_per_wave, wave_size,
                      vgpr_total, sgpr_total, max_waves_hw):
    # VGPRs consumed by one wave
    vgpr_per_wave = vgpr_per_lane * wave_size
    # maximum waves due to VGPR budget
    max_waves_vgpr = vgpr_total // vgpr_per_wave if vgpr_per_wave>0 else max_waves_hw
    # maximum waves due to SGPR budget
    max_waves_sgpr = sgpr_total // sgpr_per_wave if sgpr_per_wave>0 else max_waves_hw
    # final allocated waves per CU
    allocated_waves = min(max_waves_vgpr, max_waves_sgpr, max_waves_hw)
    occupancy = allocated_waves / float(max_waves_hw)
    return allocated_waves, occupancy

# Example: query device for real numbers (use rocminfo), then plug values here.
if __name__ == "__main__":
    # example placeholders - replace with rocminfo-derived values
    allocated, occ = compute_occupancy(vgpr_per_lane=8, sgpr_per_wave=16, wave_size=64,
                                       vgpr_total=2048, sgpr_total=800, max_waves_hw=10)
    print(f"Allocated waves per CU: {allocated}, Occupancy: {occ:.2%}")