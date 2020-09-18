//
// Created by Corrado Mio on 18/09/2020.
//

#ifndef HLSEXT_KEYS_H
#define HLSEXT_KEYS_H

#pragma once

#include <vector>
#include <map>
#include <algorithm>

namespace stdx {

    template<
        typename _Key,
        typename _Tp,
        typename _Compare = std::less<_Key>
    >
    std::vector<_Key> keys(const std::map<_Key, _Tp, _Compare>& map, bool sorted=false) {
        std::vector<_Key> kvect;
        for (auto it=map.begin(); it != map.end(); it++)
            kvect.push_back(it->first);

        if (sorted)
            std::sort(kvect.begin(), kvect.end());
        return kvect;
    }

}

#endif //HLSEXT_KEYS_H
