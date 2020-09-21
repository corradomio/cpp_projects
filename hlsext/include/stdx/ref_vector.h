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
        typedef std::vector<_Tp, _Alloc> collection;
        typedef std::shared_ptr<std::vector<_Tp, _Alloc>> pointer;

        pointer ptr;

        // Constructor

        vector() { ptr = pointer(new collection()); }
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

        typename collection::iterator       begin()        { return (*ptr).begin(); }
        typename collection::iterator       end()          { return (*ptr).end(); }
        typename collection::const_iterator begin()  const { return (*ptr).begin(); }
        typename collection::const_iterator end()    const { return (*ptr).end(); }

        typename collection::const_iterator cbegin() const { return (*ptr).cbegin(); }
        typename collection::const_iterator cend()   const { return (*ptr).cend(); }

        // Capacity

        bool  empty() const { return (*ptr).empty(); }
        size_t size() const { return (*ptr).size(); }

        // Modifiers

        void clear() { (*ptr).clear(); }
        void insert(const _Tp& v) { (*ptr).push_back(v); }
        void push_back(const _Tp& v) { (*ptr).push_back(v); }

        // Pointers

        pointer     get() const { return  ptr; }
        collection& ref() const { return *ptr; }
    };
}

#endif //HLSEXT_REF_VECTOR_H
