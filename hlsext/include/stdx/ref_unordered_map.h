//
// Created by Corrado Mio on 18/09/2020.
//

#ifndef HLSEXT_REF_UNORDERED_MAP_H
#define HLSEXT_REF_UNORDERED_MAP_H

#include <memory>
#include <unordered_map>

namespace ref {

    template<
        typename _Key,
        typename _Tp,
        typename _Hash = std::hash<_Key>,
        typename _Pred = std::equal_to<_Key>,
        typename _Alloc = std::allocator<std::pair<const _Key, _Tp> >
    >
    struct unordered_map {
        std::shared_ptr<std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc>> ptr;

        // Constructor

        unordered_map() { ptr = std::shared_ptr<std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc>>(new std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc>()); }
        unordered_map(const unordered_map& m): ptr(m.ptr) { }
        ~unordered_map() { ptr = nullptr; }

        // Member functions

        unordered_map& operator =(const unordered_map& that) {
            ptr = that.ptr;
            return *this;
        }

        // Element access

              _Tp& operator[](const _Key& k)       { return (*ptr)[k]; }
        const _Tp& operator[](const _Key& k) const { return (*ptr)[k]; }

              _Tp& at(const _Key& k)       { return (*ptr).at(k); }
        const _Tp& at(const _Key& k) const { return (*ptr).at(k); }

        // Iterators

        typename std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc>::iterator       begin()       { return (*ptr).begin(); }
        typename std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc>::const_iterator begin() const { return (*ptr).begin(); }
        typename std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc>::iterator       end()         { return (*ptr).begin(); }
        typename std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc>::const_iterator end()   const { return (*ptr).begin(); }

        // Capacity

        bool  empty() const { return (*ptr).empty(); }
        size_t size() const { return (*ptr).size(); }

        // Modifiers

        void clear() { (*ptr).clear(); }
        void insert(const _Key& k, const _Tp& v) { (*ptr)[k] = v; }
        void  erase(const _Key& k) { (*ptr).erase(k); }

        // Pointers

        std::shared_ptr<std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc>> get() const { return  ptr; }
        std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc>&                 ref() const { return *ptr; }
    };

}

#endif //HLSEXT_REF_UNORDERED_MAP_H
