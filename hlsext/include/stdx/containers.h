//
// Created by Corrado Mio on 19/09/2020.
//

#ifndef CHECK_HLSEXT_MEMBEROF_H
#define CHECK_HLSEXT_MEMBEROF_H

#include <algorithm>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>


namespace stdx {

    // ----------------------------------------------------------------------
    // Keys
    // ----------------------------------------------------------------------

    template<
        typename _Key,
        typename _Tp,
        typename _Compare = std::less<_Key>
    >
    std::vector<_Key> keys(const std::map<_Key, _Tp>& map, bool sorted=false) {
        std::vector<_Key> kvect;
        for (auto it=map.begin(); it != map.end(); it++)
            kvect.push_back(it->first);

        if (sorted)
            std::sort(kvect.begin(), kvect.end());
        return kvect;
    }

    template<
        typename _Key,
        typename _Tp,
        typename _Compare = std::less<_Key>
    >
    std::vector<_Key> keys(const std::unordered_map<_Key, _Tp>& map, bool sorted=false) {
        std::vector<_Key> kvect;
        for (auto it=map.begin(); it != map.end(); it++)
            kvect.push_back(it->first);

        if (sorted)
            std::sort(kvect.begin(), kvect.end());
        return kvect;
    }

    // ----------------------------------------------------------------------
    // Containment
    // ----------------------------------------------------------------------

    template<typename _Tp, typename _Compare>
    bool contains(const std::set<_Tp, _Compare>& set, const _Tp& v) {
        return set.find(v) != set.end();
    }

    template<typename _Tp, typename _Compare>
    bool contains(const std::unordered_set<_Tp, _Compare>& set, const _Tp& v) {
        return set.find(v) != set.end();
    }

    template<typename _Key, typename _Tp>
    bool contains_key(const std::map<_Key, _Tp>& map, const _Key& k) {
        return map.find(k) != map.end();
    }

    template<typename _Key, typename _Tp>
    bool contains_key(const std::unordered_map<_Key, _Tp>& map, const _Key& k) {
        return map.find(k) != map.end();
    }

    // ----------------------------------------------------------------------
    // Inersection
    // ----------------------------------------------------------------------

    /// s1 subsetof s2
    template<typename _Set>
    bool is_subset(const _Set& s1, const _Set& s2) {
        for(auto it=s1.cbegin(); it != s1.cend(); ++it)
            if (!contains(s2, *it))
                return false;
        return true;
    }

    template<typename _Set>
    bool has_intersection(const _Set& s1, const _Set& s2){
        for(auto it=s1.cbegin(); it != s1.cend(); ++it)
            if (contains(s2, *it))
                return true;
        return false;
    }

    template<typename _Set>
    void merge(_Set& s1, const _Set& s2) {
        for(auto it = s2.cbegin(); it != s2.cend(); ++it)
            s1.insert(*it);
    }

}

#endif //CHECK_HLSEXT_MEMBEROF_H
