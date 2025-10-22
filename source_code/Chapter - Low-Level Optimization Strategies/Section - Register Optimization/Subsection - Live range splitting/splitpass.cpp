#include <vector>
#include <algorithm>
#include <utility>
#include <iostream>

// Simple interval representation (start,end are instruction indices).
struct Interval { int reg; int start; int end; int weight; };

// Inserted move (pseudo-assembly emitter will convert to target instructions).
struct Move { int from_reg; int to_reg; int at_index; };

std::vector<Interval> splitLiveRanges(std::vector<Interval> intervals,
                                      int regBudgetPerThread,
                                      int R_SM, // SM register budget
                                      std::vector<Move>& moves) {
    // Compute current register demand (peak simultaneous intervals).
    auto peak_demand = [](const std::vector<Interval>& I) {
        std::vector<std::pair<int,int>> ev; ev.reserve(2*I.size());
        for (auto &it : I) { ev.emplace_back(it.start, +1); ev.emplace_back(it.end+1, -1); }
        std::sort(ev.begin(), ev.end());
        int cur=0, peak=0;
        for (auto &e : ev) { cur += e.second; peak = std::max(peak, cur); }
        return peak;
    };

    int peak = peak_demand(intervals);
    int max_threads = R_SM / regBudgetPerThread;
    // If peak fits budget, nothing to do.
    if (peak <= regBudgetPerThread) return intervals;

    // Greedy: sort intervals by length*weight descending (most valuable targets first).
    std::sort(intervals.begin(), intervals.end(),
              [](const Interval& a, const Interval& b) {
                  return (a.end - a.start + 1) * a.weight >
                         (b.end - b.start + 1) * b.weight;
              });

    for (size_t i = 0; i < intervals.size() && peak > regBudgetPerThread; ++i) {
        Interval cur = intervals[i];
        // Choose split point at median of interval (CFG-aware heuristics could be used instead).
        int p = (cur.start + cur.end) / 2;
        if (p <= cur.start || p >= cur.end) continue;

        // Create two intervals.
        Interval a = { cur.reg, cur.start, p, cur.weight };
        Interval b = { cur.reg, p+1, cur.end, cur.weight };

        // Assign a new virtual register ID to the second part (compiler ensures uniqueness).
        int new_reg = cur.reg + 1000; // placeholder mapping for demonstration
        b.reg = new_reg;

        // Record move at split boundary: move a.reg -> new_reg before first use in b.
        moves.push_back({ a.reg, new_reg, p+1 });

        // Replace interval with a and append b.
        intervals[i] = a;
        intervals.push_back(b);

        // Recompute peak conservatively.
        peak = peak_demand(intervals);
    }
    return intervals;
}