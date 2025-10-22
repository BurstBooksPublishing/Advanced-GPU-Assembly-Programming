#!/usr/bin/env python3
# Parse assembly text and produce stats. // small, production-ready parser
import re,sys,collections

# Architectural parameters (set per-GPU)
R_MAX = 65536        # registers per SM (example placeholder)
W_MAX = 64           # max warps per SM
THREADS_PER_WARP = 32

opcode_re = re.compile(r'^\s*([A-Za-z0-9_.]+)')          # opcode at line start
reg_re = re.compile(r'\b(?:R|%r)\d+\b', re.IGNORECASE)  # matches R0 or %r0

def analyze(lines):
    opcode_counts = collections.Counter()
    reg_counts = collections.Counter()
    max_regs_per_thread = 0
    for ln in lines:
        m = opcode_re.match(ln)
        if m:
            opcode_counts[m.group(1)] += 1
        regs = set(reg_re.findall(ln))
        for r in regs:
            reg_counts[r] += 1
        # simple per-thread reg estimate: count unique registers used per (assumed) thread
        # conservative upper bound:
        cur_regs = len(regs)
        if cur_regs > max_regs_per_thread:
            max_regs_per_thread = cur_regs
    # occupancy estimate (conservative)
    r_t = max_regs_per_thread
    if r_t == 0: r_t = 1
    active_warps = min(W_MAX, R_MAX // (r_t * THREADS_PER_WARP))
    occupancy = active_warps / W_MAX
    return opcode_counts, max_regs_per_thread, occupancy

if __name__ == '__main__':
    asm = sys.stdin.read().splitlines()
    ops, regs, occ = analyze(asm)
    print("Top opcodes:", ops.most_common(10))
    print("Estimated regs/thread:", regs)
    print("Estimated occupancy:", f"{occ*100:.1f}%")