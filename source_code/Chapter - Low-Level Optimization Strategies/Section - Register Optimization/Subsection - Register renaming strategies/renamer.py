# Simple linear-scan renamer: intervals is list of (vreg, start, end)
# phys_regs is number of available physical registers.
def linear_scan_renamer(intervals, phys_regs):
    # sort by start point
    intervals = sorted(intervals, key=lambda x: x[1])
    active = []                        # list of (end, vreg, preg)
    free_list = list(range(phys_regs)) # available physical registers
    mapping = {}                       # vreg -> preg or 'spill'
    import heapq
    active_heap = []                   # min-heap by end

    for vreg, start, end in intervals:
        # expire old intervals
        while active_heap and active_heap[0][0] <= start:
            _, old_vreg, old_preg = heapq.heappop(active_heap)
            if old_preg != 'spill':
                free_list.append(old_preg)
        # allocate
        if free_list:
            preg = free_list.pop()     # assign a physical reg
            mapping[vreg] = preg
            heapq.heappush(active_heap, (end, vreg, preg))
        else:
            # no free physical register; choose victim with latest end
            # spill policy: spill the interval with the largest end
            # (simple heuristic); mark current as spilled if better
            worst = max(active_heap, key=lambda x: x[0])
            if worst[0] > end:
                # spill worst, reuse its phys reg
                active_heap.remove(worst)
                heapq.heapify(active_heap)
                _, w_vreg, w_preg = worst
                mapping[w_vreg] = 'spill'
                mapping[vreg] = w_preg
                heapq.heappush(active_heap, (end, vreg, w_preg))
            else:
                mapping[vreg] = 'spill'
    return mapping

# Example usage: small test intervals
if __name__ == "__main__":
    ivals = [('t0',0,5),('t1',1,3),('t2',2,9),('t3',4,6),('t4',7,8)]
    print(linear_scan_renamer(ivals, phys_regs=2)) # quick allocation test