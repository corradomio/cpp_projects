//
// Created by Corrado Mio on 18/09/2020.
//

#ifndef HLSEXT_REF_VECTOR_H
#define HLSEXT_REF_VECTOR_H

#include <memory>
#include <vector>

namespace ref {

    template<
        typename _Tp,
        typename _Alloc = std::allocator<_Tp>
    >
    struct vector {
        std::shared_ptr<std::vector<_Tp, _Alloc>> ptr;

        // Constructor

        vector() { ptr = std::shared_ptr<std::vector<_Tp, _Alloc>>(new std::vector<_Tp, _Alloc>()); }
        vector(const vector& v): ptr(v.ptr) { }
        ~vector() { ptr = nullptr; }

        // Member functions

        vector& operator=(const vector& v) {
            ptr = v.ptr;
            return *this;
        }

        // Element access

              _Tp& operator[](const size_t pos)       { return (*ptr)[pos]; }
        const _Tp& operator[](const size_t pos) const { return (*ptr)[pos]; }

              _Tp& at(const size_t pos)       { return (*ptr).at(pos); }
        const _Tp& at(const size_t pos) const { return (*ptr).at(pos); }

        // Iterators

        typename std::vector<_Tp, _Alloc>::iterator       begin()       { return (*ptr).begin(); }
        typename std::vector<_Tp, _Alloc>::const_iterator begin() const { return (*ptr).begin(); }
        typename std::vector<_Tp, _Alloc>::iterator       end()         { return (*ptr).begin(); }
        typename std::vector<_Tp, _Alloc>::const_iterator end()   const { return (*ptr).begin(); }

        // Capacity

        bool  empty() const { return (*ptr).empty(); }
        size_t size() const { return (*ptr).size(); }

        // Modifiers

        void clear() { (*ptr).clear(); }
        void insert(const _Tp& v) { (*ptr).push_back(v); }
        void push_back(const _Tp& v) { (*ptr).push_back(v); }

        // Pointers

        std::shared_ptr<std::vector<_Tp, _Alloc>> get() const { return  ptr; }
        std::vector<_Tp, _Alloc>&                 ref() const { return *ptr; }
    };
}

#endif //HLSEXT_REF_VECTOR_H
