//
// Created by Corrado Mio on 02/10/2020.
//

#ifndef HLSEXT_BAG_H
#define HLSEXT_BAG_H

#include <map>
#include <set>

namespace stdx {

    template <
        typename _Key,
        typename _Tp = uint16_t,
        typename _Compare = std::less<_Key>,
        typename _Alloc = std::allocator<std::pair<const _Key, _Tp> >
    >
    class bag : public std::map<_Key, _Tp, _Compare, _Alloc> {
    public:
        typedef _Key		key_type;
        typedef std::pair<const _Key, _Tp>   value_type;
        typedef _Compare    key_compare;
        typedef _Alloc		allocator_type;
        typedef std::map<_Key, _Tp, _Compare, _Alloc> parent_type;
    public:
        bag():parent_type() { }
        bag(const bag& b): parent_type(b){ }

        template <typename Iter>
        bag(const Iter& first, const Iter& last) {
            insert(first, last);
        }

        bag(std::initializer_list<value_type> l) {
            insert(l.cbegin(), l.cend());
        }

        template <typename Iter>
        bag& insert(const Iter& first, const Iter& last) {
            for (auto it=first; it != last; ++it)
                insert(*it);
            return *this;
        }

        bag& insert(const _Key& v) {
            (*(parent_type*)this)[v] = 1 +  (*(parent_type*)this)[v];
            return *this;
        }

        bag& insert(const _Key& v, _Tp n) {
            (*(parent_type*)this)[v] = n +  (*(parent_type*)this)[v];
            return *this;
        }
    };

}

#endif //HLSEXT_BAG_H
