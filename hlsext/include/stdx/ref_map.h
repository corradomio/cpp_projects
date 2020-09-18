//
// Created by Corrado Mio on 18/09/2020.
//

#ifndef HLSEXT_REF_MAP_H
#define HLSEXT_REF_MAP_H

#include <memory>
#include <map>

namespace ref {

    template<
        typename _Key,
        typename _Tp,
        typename _Compare = std::less<_Key>,
        typename _Alloc = std::allocator<std::pair<const _Key, _Tp> >
    >
    struct map {
        std::shared_ptr<std::map<_Key, _Tp, _Compare, _Alloc>> p;

        // Constructor

        map() { p = std::shared_ptr<std::map<_Key, _Tp, _Compare, _Alloc>>(new std::map<_Key, _Tp, _Compare, _Alloc>()); }
        map(const map& m): p(m.p) { }
        ~map() { p = nullptr; }

        // Member functions

        map& operator =(const map& that) {
            p = that.p;
            return *this;
        }

        // Element access

              _Tp& operator[](const _Key& k)       { return (*p)[k]; }
        const _Tp& operator[](const _Key& k) const { return (*p)[k]; }

              _Tp& at(const _Key& k)       { return (*p).at(k); }
        const _Tp& at(const _Key& k) const { return (*p).at(k); }

        // Iterators

        typename std::map<_Key, _Tp, _Compare, _Alloc>::iterator       begin()       { return (*p).begin(); }
        typename std::map<_Key, _Tp, _Compare, _Alloc>::const_iterator begin() const { return (*p).begin(); }
        typename std::map<_Key, _Tp, _Compare, _Alloc>::iterator       end()         { return (*p).begin(); }
        typename std::map<_Key, _Tp, _Compare, _Alloc>::const_iterator end()   const { return (*p).begin(); }

        // Capacity

        bool  empty() const { return (*p).empty(); }
        size_t size() const { return (*p).size(); }

        // Modifiers

        void clear() { (*p).clear(); }

        void insert(const _Key& k, const _Tp& v) { (*p)[k] = v; }

        // Pointers

        std::shared_ptr<std::map<_Key, _Tp, _Compare, _Alloc>> get() const { return p; }
        std::map<_Key, _Tp, _Compare, _Alloc>&                 ref() const { return *p; }
    };

}

#endif //HLSEXT_REF_MAP_H
