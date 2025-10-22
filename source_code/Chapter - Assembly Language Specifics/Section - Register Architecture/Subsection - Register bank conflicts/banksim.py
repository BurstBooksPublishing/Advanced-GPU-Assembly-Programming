# warp_bank_sim.py -- simple model to quantify bank conflicts
from collections import Counter
import math

def bank_for_reg(reg_idx, B):
    return reg_idx % B  # simple modulo mapping

def cycles_for_instruction(reg_map, B=8, P=1, W=32):
    # reg_map: list of lists: reg_map[t] = [r_t,1, r_t,2, ...] for lane t
    counts = Counter()
    for t in range(W):
        for r in reg_map[t]:
            counts[bank_for_reg(r, B)] += 1
    # compute cycles per bank
    cycles = max((math.ceil(counts[b] / P) for b in range(B)), default=0)
    return cycles

# Example: each lane reads its own reg and a neighbor reg (creates clustering)
W = 32; B = 8; P = 1
reg_map = []
for t in range(W):
    # naive allocator: reg = t, neighbor = (t+1)%W -> possible bank clustering
    reg_map.append([t, (t+1) % W])
print("cycles:", cycles_for_instruction(reg_map, B=B, P=P, W=W))