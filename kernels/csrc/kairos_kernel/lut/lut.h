#ifndef LUT_H
#define LUT_H

#include <vector>
#include <algorithm>
#include <cstdint>

class LUT {
private:
    std::vector<int> seg_list;
    std::vector<int> best_comp;
    // Record the recent ask.
    int pre_ask = -1, pre_res;
    
public:
    LUT() {
        seg_list.reserve(32);
        seg_list.push_back(0);
        best_comp.reserve(32);
    }
    
    int get(int m) {
        if (m == pre_ask) return pre_res;
        pre_ask = m;
        if (m >= seg_list.back()) {
            pre_res = best_comp.back();
            return pre_res;
        }
        auto it = std::upper_bound(seg_list.begin(), seg_list.end(), m);
        size_t index = std::distance(seg_list.begin(), it) - 1;
        pre_res = best_comp[index];
        return pre_res;
    }
    
    void record(int r, int comp_type) {
        seg_list.push_back(r + 1); // to make a left closed right open interval.
        best_comp.push_back(comp_type);
    }
    
    void shrink_to_fit() {
        seg_list.shrink_to_fit();
        best_comp.shrink_to_fit();
    }
};
#endif // LUT_H