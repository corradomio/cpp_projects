//
// Created by Corrado Mio on 18/09/2020.
//

#ifndef HLSEXT_REF_MAP_H
#define HLSEXT_REF_MAP_H

#include <memory>
#include <map>

namespace ref {

    template<typename K, typename V>
    struct map {
        std::shared_ptr<std::map<K, V>> p;

        map() { p = new std::map<K, V>(); }
       ~map() { p = nullptr; }

        V& operator[](const K& k) {
            return (*p)[k];
        }

        const V& operator[](const K& k) const {
            return (*p)[k];
        }

        map& operator =(const map& that) {
            p = that.p;
            return *this;
        }
    };

}

#endif //HLSEXT_REF_MAP_H
