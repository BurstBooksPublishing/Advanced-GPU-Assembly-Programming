# Simple spill optimizer: assign temporaries to shared memory slots (fast) when possible,
# otherwise assign to global-local memory. Emits load/store pseudo-instructions.
# Inputs: live_ranges: dict(temp -> (start, end)), shared_capacity (slots per block)
def allocate_spills(live_ranges, shared_capacity, warp_size=32):
    # sort temps by decreasing live-range length (heuristic)
    temps = sorted(live_ranges.keys(), key=lambda t: live_ranges[t][1]-live_ranges[t][0], reverse=True)
    shared_map = {}   # temp -> shared_slot index
    global_map = {}   # temp -> global_slot index
    next_shared = 0
    next_global = 0
    for t in temps:
        length = live_ranges[t][1]-live_ranges[t][0]
        # prefer short-lived to global spills; try rematerialization threshold
        if next_shared < shared_capacity and length <= 128:
            shared_map[t] = next_shared; next_shared += 1
        else:
            global_map[t] = next_global; next_global += 1
    # emitter: for each use, if temp in shared_map emit ld.shared/st.shared
    def emit_spill_code(instr_list):
        emitted = []
        for instr in instr_list:
            for t in instr['defs']:
                if t in global_map:
                    # store before a long gap; use vector store per-warp (coalesced)
                    emitted.append(f"st.global.v4.u32 [g_spill+{global_map[t]}+lane*4], %r{t} // global coalesced")
                elif t in shared_map:
                    emitted.append(f"st.shared.u32 [s_spill+{shared_map[t]}+lane], %r{t} // fast per-block")
            emitted.append(instr['asm'])
            for t in instr['uses']:
                if t in global_map:
                    emitted.append(f"ld.global.v4.u32 %r{t}, [g_spill+{global_map[t]}+lane*4] // coalesced load")
                elif t in shared_map:
                    emitted.append(f"ld.shared.u32 %r{t}, [s_spill+{shared_map[t]}+lane] // fast fill")
        return emitted
    return shared_map, global_map, emit_spill_code