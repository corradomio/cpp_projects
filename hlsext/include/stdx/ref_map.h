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
        typedef std::map<_Key, _Tp, _Compare, _Alloc> collection;
        typedef std::shared_ptr<std::map<_Key, _Tp, _Compare, _Alloc>> pointer;

        pointer ptr;

        // Constructor

        map() { ptr = pointer(new collection()); }
        map(const map& m): ptr(m.ptr) { }
        ~map() { ptr = nullptr; }

        // Member functions

        map& operator =(const map& that) {
            ptr = that.ptr;
            return *this;
        }

        // Element access

              _Tp& operator[](const _Key& k)       { return (*ptr)[k]; }
        const _Tp& operator[](const _Key& k) const { return (*ptr)[k]; }

              _Tp& at(const _Key& k)       { return (*ptr).at(k); }
        const _Tp& at(const _Key& k) const { return (*ptr).at(k); }

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
        void insert(const _Key& k, const _Tp& v) { (*ptr)[k] = v; }
        void  erase(const _Key& k) { (*ptr).erase(k); }

        // Pointers

        pointer     get() const { return  ptr; }
        collection& ref() const { return *ptr; }
    };

}

#endif //HLSEXT_REF_MAP_H
