//
// Created by Corrado Mio on 15/06/2024.
//

#ifndef STDX_NUMBERS_H
#define STDX_NUMBERS_H

namespace stdx::number::i128::cns {

    typedef unsigned __int128 iset_t;

    constexpr iset_t ONE = iset_t(1);

    int ihighbit(iset_t S);
    int icard(iset_t S);

    /// integer value into combinatorial number system
    iset_t ilexset(iset_t L, int n);

    /// combinarorial number system's integer to integer
    iset_t ilexidx(iset_t S, int n);
}

#endif //STDX_NUMBERS_H
