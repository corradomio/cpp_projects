//
// Created by Corrado Mio (Local) on 23/04/2021.
//

#ifndef GENERIC_HASH_HPP
#define GENERIC_HASH_HPP

namespace std {

    template<typename _Tp1, typename... _TpRest>
    struct hash_combine {
        size_t operator() () {
            return 0;
        }
        size_t operator() (_Tp1 value1, _TpRest... rest) {
            return (std::hash<_Tp1>{}(value1) << 1) ^ (std::hash_combine<_TpRest...>{}(rest...));
        }
    };

    template<typename T1, typename T2>
        struct hash<pair<T1,T2>>: public __hash_base<size_t, pair<T1,T2>> {
            //size_t operator()(const pair<T1,T2>& p) const {
            //    hash<T1> h1;
            //    hash<T2> h2;
            //    return (h1(p.first) << 0) ^  (h2(p.second) << 16);
            //}
            size_t operator()(const pair<T1,T2>& p) {
                return std::hash_combine<T1,T2>{}(p.first, p.second);
            }
        };

}

#endif // GENERIC_HASH_HPP
