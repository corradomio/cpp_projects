//
// Created by Corrado Mio (Local) on 23/04/2021.
//

#ifndef GENERIC_HASH_HPP
#define GENERIC_HASH_HPP

namespace std {

    template<typename T1, typename T2>
    struct hash<pair<T1,T2>>: public __hash_base<size_t, pair<T1,T2>> {
        size_t operator()(const pair<T1,T2>& p) const {
            hash<T1> h1;
            hash<T2> h2;
            return (h1(p.first) << 0) ^  (h2(p.second) << 16);
        }
    };
}

#endif // GENERIC_HASH_HPP
