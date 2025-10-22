# Simple scheduler: instructions are tuples (id, cls, latency, dst_regs, src_regs)
# cls in {'V','S','M'} for vector, scalar, memory. Dependencies on src_regs.
from collections import deque, defaultdict

def schedule(instrs, vgpr_budget, I_v=1, I_s=1, I_m=1):
    ready = deque(instrs)            # naive ready queue; real scheduler would be priority-based
    pending = []                     # instructions in flight (latency countdown)
    reg_live = set()                 # current live VGPRs
    cycle = 0
    bundles = []                     # list of per-cycle bundles
    while ready or pending:
        # retire finished instrs
        pending = [(inst, t-1) for inst,t in pending if t-1>0]
        # update live regs conservatively (no precise live-range analysis here)
        # build bundle by trying to pick at most one of each class
        slot_v = slot_s = slot_m = None
        used_regs = set()
        for i, inst in enumerate(list(ready)):
            _, cls, lat, dst, src = inst
            # dependency check: ensure src not produced by earlier ready insts
            if any(r in used_regs for r in src): continue
            # resource check
            if cls=='V' and slot_v is None:
                # register budget check
                if len(reg_live.union(dst))<=vgpr_budget:
                    slot_v = inst; used_regs |= set(dst); ready.remove(inst)
            elif cls=='S' and slot_s is None:
                slot_s = inst; used_regs |= set(dst); ready.remove(inst)
            elif cls=='M' and slot_m is None:
                slot_m = inst; used_regs |= set(dst); ready.remove(inst)
        bundle = [x for x in (slot_v, slot_s, slot_m) if x]
        # if no instruction selected, advance cycle for pending latency
        if not bundle:
            # if nothing can issue now, stall one cycle
            bundles.append([])
            cycle += 1
            continue
        # issue selected instructions
        for inst in bundle:
            _, cls, lat, dst, src = inst
            pending.append((inst, lat))
            reg_live |= set(dst)
        bundles.append(bundle)
        cycle += 1
    return bundles

# Example: chain of operations: memory loads with latency 200, some compute F32 ops.
instrs = [
  ('L0','M',200, ('v10',),()), ('A0','V',1, ('v11',),('v10','v12')),
  ('A1','V',1, ('v13',),('v11','v14')), ('S0','S',1, ('s0',),()),
  ('L1','M',200, ('v15',),()), ('A2','V',1, ('v16',),('v15',))
]
bundles = schedule(instrs, vgpr_budget=256)
print("Cycles:", len(bundles), "Issued bundles sample:", bundles[:10])  # quick inspection