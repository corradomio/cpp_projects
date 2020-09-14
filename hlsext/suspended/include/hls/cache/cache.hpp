//
// Created by Corrado Mio on 20/03/2016.
//

#ifndef HLS_CACHE_HASHMAP_HPP
#define HLS_CACHE_HASHMAP_HPP
#include <functional>

namespace hls {
namespace cache {

    class random_strategy;

    template<
        class _Key,
        class _Tp,
        class _Hash = std::hash<_Key>,
        class _Pred = std::equal_to<_Key>,
        class _Strategy = random_strategy
    >
    class cache {
        typedef size_t size_type;
        typedef size_t interval_type;
    public:
        cache();

        void put(const _Key& key, const _Tp& value);
        void put(const _Key& key, const _Tp& value, interval_type millis);

        bool has(const _Key& key) const;
        const _Tp& get(const _Key& key);
    };

}}

#endif //HLS_CACHE_HASHMAP_HPP
