//
// Created by Corrado Mio on 18/09/2020.
//

#ifndef HLSEXT_REF_SET_H
#define HLSEXT_REF_SET_H

#include <memory>
#include <set>

namespace ref {

    template<
        typename _Key,
        typename _Compare = std::less<_Key>,
        typename _Alloc = std::allocator<_Key>
    >
    struct set {
        typedef std::set<_Key, _Compare, _Alloc> collection;
        typedef std::shared_ptr<std::set<_Key, _Compare, _Alloc>> pointer;

        std::shared_ptr<std::set<_Key, _Compare, _Alloc>> ptr;

        // Constructor

        set() { ptr = std::shared_ptr<std::set<_Key, _Compare, _Alloc>>(new std::set<_Key, _Compare, _Alloc>()); }
        set(const set& v): ptr(v.ptr) { }
        ~set() { ptr = nullptr; }

        // Member functions

        set& operator=(const set& v) {
            ptr = v.ptr;
            return *this;
        }

        // Element access

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
        void insert(const _Key& v) { (*ptr).insert(v); }
        void  erase(const _Key& v) { (*ptr).erase(v); }

        // Pointers

        std::shared_ptr<std::set<_Key, _Compare, _Alloc>> get() const { return  ptr; }
        std::set<_Key, _Compare, _Alloc>&                 ref() const { return *ptr; }
    };
}

#endif //HLSEXT_REF_SET_H
