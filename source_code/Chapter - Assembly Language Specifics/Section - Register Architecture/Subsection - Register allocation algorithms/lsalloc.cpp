#include <vector>
#include <algorithm>
// Interval: [start,end), weight: heuristic priority.
struct Interval { int id, start, end; double weight; };
// Assign registers 0..K-1, spill to memory id = -1.
std::vector<int> linear_scan_alloc(std::vector<Interval> intervals, int K) {
    std::sort(intervals.begin(), intervals.end(),
              [](auto &a, auto &b){ return a.start < b.start; });
    std::vector<int> reg_of(intervals.size(), -1);
    std::vector<std::pair<int,int>> active; // (end, reg)
    std::vector<int> free_regs(K);
    for(int i=0;i<K;++i) free_regs[i]=i;
    for(size_t i=0;i<intervals.size();++i){
        // expire old intervals
        for(auto it=active.begin(); it!=active.end();){
            if(it->first <= intervals[i].start){ free_regs.push_back(it->second); it = active.erase(it); }
            else ++it;
        }
        if(!free_regs.empty()){
            // pick lowest reg (bank-aware choice could map reg%bank)
            int r = free_regs.back(); free_regs.pop_back();
            reg_of[intervals[i].id] = r;
            active.emplace_back(intervals[i].end, r);
            std::sort(active.begin(), active.end()); // small active set typical on GPUs
        } else {
            // spill: pick active interval with smallest weight to spill
            auto victim_it = active.end();
            double min_w = 1e300; size_t idx = -1;
            for(size_t j=0;j<active.size();++j){
                if(intervals[active[j].second].weight < min_w){
                    min_w = intervals[active[j].second].weight;
                    idx = j;
                }
            }
            victim_it = active.begin()+idx;
            int free_reg = victim_it->second;
            reg_of[intervals[i].id] = free_reg;
            // mark victim spilled (not shown: generate spill code and split interval)
            active.erase(victim_it);
            active.emplace_back(intervals[i].end, free_reg);
            std::sort(active.begin(), active.end());
        }
    }
    return reg_of;
}