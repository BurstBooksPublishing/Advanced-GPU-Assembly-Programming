# inputs: instrs = list of dict {id, reads:set, writes:set, is_barrier:bool}
# output: reordered list preserving dependencies and barriers
def reorder_instructions(instrs):
    scheduled = []
    unscheduled = instrs.copy()
    # dependency graph: can't schedule B before A if A writes a reg B reads
    def has_dependency(a,b):
        return bool(a['writes'] & b['reads']) or bool(a['writes'] & b['writes'])
    # barriers partition scheduling windows
    while unscheduled:
        # pick window up to next barrier (inclusive)
        window = []
        while unscheduled:
            ins = unscheduled.pop(0)
            window.append(ins)
            if ins.get('is_barrier'):
                break
        # greedy schedule within window: pick an instruction with no unscheduled deps
        ready = [i for i in window if not any(has_dependency(j,i) for j in window)]
        while window:
            # pick the instruction that, when scheduled, increases distance for future consumers
            # metric: number of consumers (heuristic)
            pick = max(ready, key=lambda x: sum(1 for j in window if x['writes'] & j['reads']))
            scheduled.append(pick)
            window.remove(pick)
            # recompute ready set
            ready = [i for i in window if not any(has_dependency(j,i) for j in window)]
    return scheduled
# This pass should be integrated into codegen where read/write sets are precise.