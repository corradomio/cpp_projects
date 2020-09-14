//
// Created by Corrado Mio on 20/03/2016.
//

#ifndef HLS_COLLECTION_HASHMAP_HPP
#define HLS_COLLECTION_HASHMAP_HPP

#include <unordered_map>

namespace hls {
namespace collection {

    template<
        class _Key,
        class _Tp,
        class _Hash = std::hash<_Key>,
        class _Pred = std::equal_to<_Key>,
        class _Alloc = std::allocator<std::pair<const _Key, _Tp> >
    >
    class hashmap : public std::unordered_map<_Key, _Tp> {
        typedef std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc> umap;
    public:
        hashmap() { }

        void put(const _Key& key, const _Tp& value) {
            umap::insert(std::make_pair(key,value));
        }

        bool has(const _Key& key) const { return count(key) > 0; }

        const _Tp& get(const _Key& key) { return umap::at(key); }
    };

}}

#endif //HLS_COLLECTION_HASHMAP_HPP
