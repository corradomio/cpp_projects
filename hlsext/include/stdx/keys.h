//
// Created by Corrado Mio on 18/09/2020.
//

#ifndef HLSEXT_KEYS_H
#define HLSEXT_KEYS_H

#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>

namespace stdx {

    template<typename K, typename V>
    std::vector<K> keys(const std::map<K, V>& map, bool sorted=false) {
        std::vector<K> kvect;
        for (auto it=map.begin(); it != map.end(); it++)
            kvect.push_back(it->first);

        if (sorted)
            std::sort(kvect.begin(), kvect.end());
        return kvect;
    }

    template<typename K, typename V>
    std::vector<K> keys(const std::unordered_map<K, V>& map, bool sorted=false) {
        std::vector<K> kvect;
        for (auto it=map.begin(); it != map.end(); it++)
            kvect.push_back(it->first);

        if (sorted)
            std::sort(kvect.begin(), kvect.end());
        return kvect;
    }
}

#endif //HLSEXT_KEYS_H
