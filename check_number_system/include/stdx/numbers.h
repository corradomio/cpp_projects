//
// Created by Corrado Mio on 15/06/2024.
//

#ifndef STDX_NUMBERS_H
#define STDX_NUMBERS_H

namespace stdx::number::cns {

    typedef long iset_t;

    int ihighbit(iset_t S);
    int icard(iset_t S);

    /// integer value into combinatorial number system
    long ilexset(iset_t L, int n);

    /// combinarorial number system's integer to integer
    iset_t ilexidx(iset_t S, int n);
}

#endif //STDX_NUMBERS_H
