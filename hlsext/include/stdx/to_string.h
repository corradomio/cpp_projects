//
// Created by Corrado Mio (Local) on 18/06/2021.
//

#ifndef INFSIM_TO_STRING_H
#define INFSIM_TO_STRING_H
#include <unordered_set>

namespace std {

    template<typename V>
    std::string to_string(const std::unordered_set<V> & c) {
        if (c.empty()) return "{ }";
        std::string s = "{ ";
        bool rest = false;
        for (auto it = c.cbegin(); it != c.cend(); ++it) {
            if (rest) s += ", ";
            s += std::to_string(*it);
            rest = true;
        }
        s += " }";
        return s;
    }
}

#endif //INFSIM_TO_STRING_H
