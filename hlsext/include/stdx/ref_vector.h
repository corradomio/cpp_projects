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
        std::shared_ptr<std::vector<_Tp, _Alloc>> p;

        // Constructor

        vector() { p = std::shared_ptr<std::vector<_Tp, _Alloc>>(new std::vector<_Tp, _Alloc>()); }
        vector(const vector& v): p(v.p) { }
        ~vector() { p = nullptr; }

        // Member functions

        vector& operator=(const vector& v) {
            p = v.p;
            return *this;
        }

        // Element access

              _Tp& operator[](const size_t pos)       { return (*p)[pos]; }
        const _Tp& operator[](const size_t pos) const { return (*p)[pos]; }

              _Tp& at(const size_t pos)       { return (*p).at(pos); }
        const _Tp& at(const size_t pos) const { return (*p).at(pos); }

        // Iterators

        typename std::vector<_Tp, _Alloc>::iterator       begin()       { return (*p).begin(); }
        typename std::vector<_Tp, _Alloc>::const_iterator begin() const { return (*p).begin(); }
        typename std::vector<_Tp, _Alloc>::iterator       end()         { return (*p).begin(); }
        typename std::vector<_Tp, _Alloc>::const_iterator end()   const { return (*p).begin(); }

        // Capacity

        bool  empty() const { return (*p).empty(); }
        size_t size() const { return (*p).size(); }

        // Modifiers

        void clear() { (*p).clear(); }

        void    insert(const _Tp& v) { (*p).push_back(v); }
        void push_back(const _Tp& v) { (*p).push_back(v); }

        // Pointers

        std::shared_ptr<std::vector<_Tp, _Alloc>> get() const { return p; }
        std::vector<_Tp, _Alloc>&                 ref() const { return *p; }
    };
}

#endif //HLSEXT_REF_VECTOR_H
