def classify_bottleneck(metrics):
    # metrics: dict with keys like 'sm_active_cycles','drain_bytes','dram_bw',
    # 'peak_dram_bw','flop_count','peak_flop','active_warps','max_warps'
    I = metrics['flop_count'] / max(metrics['drain_bytes'], 1)  # arithmetic intensity
    roof = min(metrics['peak_flop'], metrics['peak_dram_bw'] * I)
    mem_bound = (metrics['dram_bw'] / metrics['peak_dram_bw'] > 0.9) and (roof < 0.95 * metrics['peak_flop'])
    occupancy = metrics['active_warps'] / metrics['max_warps']
    compute_util = metrics['issued_flops'] / max(metrics['peak_flop'] * (metrics['sm_active_cycles']/metrics['cycle_hz']), 1)
    # simple rule set
    if mem_bound:
        return 'Memory bandwidth bound'
    if occupancy < 0.25 and compute_util < 0.5:
        return 'Low occupancy; likely latency bound (increase concurrency)'
    if compute_util > 0.85:
        return 'Compute bound (optimize ALU/tensor use, instruction mix)'
    if metrics['l2_miss_rate'] > 0.2:
        return 'Cache miss bound (optimize access pattern, coalesce)'
    return 'Mixed/IO bound (investigate shared memory, atomics, pipeline stalls)'