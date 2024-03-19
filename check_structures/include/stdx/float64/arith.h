//
// Created by Corrado Mio on 08/03/2024.
//
#include <cmath>

#ifndef STDX_FLOAT64_ARITH_H
#define STDX_FLOAT64_ARITH_H

namespace stdx::float64 {

    typedef double real_t;

    inline real_t neg(real_t x)             { return -x;    }
    inline real_t  sq(real_t x)             { return x * x; }
    inline real_t sum(real_t x, real_t y)   { return x + y; }
    inline real_t sub(real_t x, real_t y)   { return x - y; }
    inline real_t mul(real_t x, real_t y)   { return x * y; }
    inline real_t div(real_t x, real_t y)   { return x / y; }
    inline real_t sqsub(real_t x, real_t y) { return sq(x-y); }
    inline real_t eps(real_t x, int n=3)    { return x*std::pow(2.,-16+n); }

}

#endif //STDX_FLOAT64_ARITH_H
