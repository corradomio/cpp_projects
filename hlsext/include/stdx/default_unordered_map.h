//
// Created by Corrado Mio on 25/09/2020.
//

#include <unordered_map>

#ifndef CHECK_TRACKS_DEFAULT_UNORDERED_MAP_H
#define CHECK_TRACKS_DEFAULT_UNORDERED_MAP_H

namespace stdx {

    template <typename _Key, typename _Tp,
        typename _Hash = std::hash<_Key>,
        typename _Pred = std::equal_to<_Key>,
        typename _Alloc = std::allocator<std::pair<const _Key, _Tp>>>
    class default_unordered_map : public std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc> {
        const _Tp defval;

        using um = std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc>;
    public:
        default_unordered_map() { }
        default_unordered_map(const _Tp& v): defval(v) { }

        const _Tp& at(const _Key& key) const {
            if (((um*)this)->find(key) == ((um*)this)->end())
                return defval;
            else
                return ((um*)this)->at(key);
        }

    };

    //template <typename _Key, typename _Tp,
    //    typename _Hash = std::hash<_Key>,
    //    typename _Pred = std::equal_to<_Key>,
    //    typename _Alloc = std::allocator<std::pair<const _Key, _Tp>>>
    //class default_unordered_map {
    //    std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc> umap;
    //    const _Tp defval;
    //public:
    //    typedef _Key    key_type;
    //    typedef _Tp     value_type;
    //    typedef _Hash   hasher;
    //    typedef _Pred   key_equal;
    //    typedef _Alloc  allocator_type;
    //
    //public:
    //    default_unordered_map(): defval(_Tp()) { }
    //    default_unordered_map(const _Tp& v): defval(v) { }
    //
    //    _Tp& operator[](const _Key& key) {
    //        return umap[key];
    //    }
    //
    //    const _Tp& operator[](const _Key& key) const {
    //        return umap[key];
    //    }
    //
    //    const _Tp& get(const _Key& key) const {
    //        if (umap.find(key) == umap.end())
    //            return defval;
    //        else
    //            return umap[key];
    //    }
    //};

}

using namespace std;

#endif //CHECK_TRACKS_DEFAULT_UNORDERED_MAP_H
