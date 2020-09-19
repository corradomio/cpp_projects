//
// Created by Corrado Mio on 18/09/2020.
//

#ifndef HLSEXT_REF_UNORDERED_SET_H
#define HLSEXT_REF_UNORDERED_SET_H

#include <memory>
#include <unordered_set>

namespace ref {

    template<typename _Value,
        typename _Hash = std::hash<_Value>,
        typename _Pred = std::equal_to<_Value>,
        typename _Alloc = std::allocator<_Value>
    >
    struct unordered_set {
        std::shared_ptr<std::unordered_set<_Value, _Hash, _Pred, _Alloc>> ptr;

        // Constructor

        unordered_set() { ptr = std::shared_ptr<std::unordered_set<_Value, _Hash, _Pred, _Alloc>>(new std::unordered_set<_Value, _Hash, _Pred, _Alloc>()); }
        unordered_set(const unordered_set& v): ptr(v.ptr) { }
        ~unordered_set() { ptr = nullptr; }

        // Member functions

        unordered_set& operator=(const unordered_set& v) {
            ptr = v.ptr;
            return *this;
        }

        // Element access

        // Iterators

        typename std::unordered_set<_Value, _Hash, _Pred, _Alloc>::iterator       begin()       { return (*ptr).begin(); }
        typename std::unordered_set<_Value, _Hash, _Pred, _Alloc>::const_iterator begin() const { return (*ptr).begin(); }
        typename std::unordered_set<_Value, _Hash, _Pred, _Alloc>::iterator       end()         { return (*ptr).end(); }
        typename std::unordered_set<_Value, _Hash, _Pred, _Alloc>::const_iterator end()   const { return (*ptr).end(); }

        // Capacity

        bool  empty() const { return (*ptr).empty(); }
        size_t size() const { return (*ptr).size(); }

        // Modifiers

        void clear() { (*ptr).clear(); }
        void insert(const _Value& v) { (*ptr).insert(v); }
        void  erase(const _Value& v) { (*ptr).erase(v); }

        // Pointers

        std::shared_ptr<std::unordered_set<_Value, _Hash, _Pred, _Alloc>> get() const { return  ptr; }
        std::unordered_set<_Value, _Hash, _Pred, _Alloc>&                 ref() const { return *ptr; }
    };
}

#endif //HLSEXT_REF_UNORDERED_SET_H
