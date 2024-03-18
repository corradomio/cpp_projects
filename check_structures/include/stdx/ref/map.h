//
// Created by Corrado Mio on 18/03/2024.
//
// template<
//     class Key,
//     class T,
//     class Compare = std::less<Key>,
//     class Allocator = std::allocator<std::pair<const Key, T>>
// > class map;
//
//  Member functions
//      operator=
//
//  Access
//      at()
//      operator []
//
//  Iterators
//      begin/cbegin/enc/cend/rbegin/rcbegin/rend/rcend
//
//  Capacity
//      empty
//      size
//      max_size
//
//  Modifiers
//      clear
//      insert
//      insert_range
//      insert_or_assign
//      emplace
//      emplace_hint
//      try_emplace
//      erase
//      swap
//      extract
//      merge
//
//  Lookup
//      count
//      find
//      contains
//      equal_range
//      lower_bound
//      upper_bound
//




#ifndef STDX_REF_MAP_H
#define STDX_REF_MAP_H

#include "../language.h"
#include <cstdlib>
#include <map>

// map<

namespace stdx::ref {

    template<typename Key, typename T>
    class map {
        size_t *pref;
        std::map<Key, T> *pmap;

        void add_ref() { (*pref)++; }
        void release() {
            if (0 == --(*pref)) {
                delete pref;
                delete pmap;
            }
        }
    public:
        using key_type = Key;
        using mapped_type = T;
        using value_type = std::pair<const Key, T>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using reference = value_type&;
    public:
        map(): pref(new size_t(0)), pmap(new std::map<Key, T>()) {
            self.add_ref();
        }

        map(const map& m): pref(m.pref), pmap(m.pmap) {
            self.add_ref();
        }

        ~map() {
            self.release();
        }

        map& operator =(const map& m) {
            m.add_ref();
            self.release();
            pref = m.pref;
            pmap = m.pmap;
            return self;
        }

              T& at( const Key& key )       { return (*pmap).at(key); }
        const T& at( const Key& key ) const { return (*pmap).at(key); }

        T& operator[](const Key& key) {
            return (*pmap)[key];
        }

        size_type size() const { return (*pmap).size(); }
        size_type max_size() const { return (*pmap).max_size(); }
    };

}

#endif //STDX_REF_MAP_H
