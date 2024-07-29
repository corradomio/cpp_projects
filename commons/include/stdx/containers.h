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

/*
 *
 *      https://en.cppreference.com/w/cpp/container
 *
 * Sequence containers
 *      array vector deque  forward_list list valarray
 *
 * Associative containers
 *      set map multiset multimap
 *
 * Unordered associative containers
 *      [unordered_] set map multiset multimap
 *
 * Container adaptors
 *      stack queue priority_queue
 *      [flat_] set map multiset multimap
 *
 * Insertion
 *      std::set::insert, std::map::emplace, std::vector::push_back, std::deque::push_front
 *
 * Erasure
 *       std::set::erase, std::vector::pop_back, std::deque::pop_front, std::map::clear
 */

namespace stdx {

    // ----------------------------------------------------------------------
    // add / extend
    // ----------------------------------------------------------------------
    // add an element to a collection

    template<typename T>
    inline std::vector<T>& add(std::vector<T>& vec, const T& val) {
        vec.push_back(val);
        return vec;
    }

    template<typename T>
    inline std::set<T>& add(std::set<T>& set, const T& elt) {
        set.insert(elt);
        return set;
    }

    template<typename T>
    inline std::unordered_set<T>& add(std::unordered_set<T>& set, const T& elt) {
        set.insert(elt);
        return set;
    }

    // ----------------------------------------------------------------------
    // copy
    // ----------------------------------------------------------------------

    /// unordered_set <- vector
    template<typename _Tp>
    void copy(std::unordered_set<_Tp>& tcoll, const std::vector<_Tp>& fcoll) {
        for(auto it = fcoll.begin(); it != fcoll.end(); ++it) {
            tcoll.emplace(*it);
        }
    }

    /// unordered_set <- unordered_set
    template<typename _Tp>
    void copy(std::unordered_set<_Tp>& tcoll, const std::unordered_set<_Tp>& fcoll) {
        for(auto it = fcoll.begin(); it != fcoll.end(); ++it) {
            tcoll.emplace(*it);
        }
    }

    // ----------------------------------------------------------------------
    // Keys
    // ----------------------------------------------------------------------

    /// keys <= keys(map)
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

    /// keys <- keys(unordered_map)
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

    /// contains(collection, value)
    template<typename Collection>
    bool contains(const Collection& coll, const typename Collection::value_type& v) {
        return coll.find(v) != coll.end();
    }

    /// contains_key(collection, key)
    template<typename Dictionary>
    bool contains_key(const Dictionary& dict, const typename Dictionary::key_type& k) {
        return dict.find(k) != dict.end();
    }

    // ----------------------------------------------------------------------
    // Intersection
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

    // ----------------------------------------------------------------------
    // End
    // ----------------------------------------------------------------------

}

#endif //CHECK_HLSEXT_MEMBEROF_H
