//
// Created by Corrado Mio on 02/10/2020.
//

#ifndef HLSEXT_BAG_H
#define HLSEXT_BAG_H

#include <unordered_map>

namespace stdx {

    template <
        typename _Key,
        typename _Tp = uint16_t,
        typename _Hash = std::hash<_Key>,
        typename _Pred = std::equal_to<_Key>,
        typename _Alloc = std::allocator<std::pair<const _Key, _Tp>>
    >
    class unordered_bag : public std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc> {
    public:
        typedef _Key			key_type;
        typedef std::pair<const _Key, _Tp>   value_type;
        typedef _Hash       hasher;
        typedef _Alloc		allocator_type;
        typedef std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc> parent_type;
    public:
        unordered_bag():parent_type() { }
        unordered_bag(const unordered_bag& b): parent_type(b){ }

        template <typename Iter>
        unordered_bag(const Iter& first, const Iter& last) {
            insert(first, last);
        }

        unordered_bag(std::initializer_list<value_type> l) {
            insert(l.cbegin(), l.cend());
        }

        template <typename Iter>
        unordered_bag& insert(const Iter& first, const Iter& last) {
            for (auto it=first; it != last; ++it)
                insert(*it);
            return *this;
        }

        unordered_bag& insert(const _Key& v) {
            (*(parent_type*)this)[v] = 1 +  (*(parent_type*)this)[v];
            return *this;
        }

        unordered_bag& insert(const _Key& v, uint32_t n) {
            (*(parent_type*)this)[v] = n +  (*(parent_type*)this)[v];
            return *this;
        }
    };

}

#endif //HLSEXT_BAG_H
