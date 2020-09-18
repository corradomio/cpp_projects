//
// Created by Corrado Mio on 18/09/2020.
//

#ifndef HLSEXT_REF_UNORDERED_SET_H
#define HLSEXT_REF_UNORDERED_SET_H

#include <memory>
#include <unordered_set>

namespace ref {

    template<typename K, typename V>
    struct unordered_set {
        std::shared_ptr<std::unordered_set<K, V>> p;

        unordered_set() { p = new std::unordered_set<K, V>(); }
        ~unordered_set() { p = nullptr; }

        V& operator[](const K& k) {
            return (*p)[k];
        }

        const V& operator[](const K& k) const {
            return (*p)[k];
        }

        unordered_set& operator =(const unordered_set& that) {
            p = that.p;
            return *this;
        }
    };

}

#endif //HLSEXT_REF_UNORDERED_SET_H
