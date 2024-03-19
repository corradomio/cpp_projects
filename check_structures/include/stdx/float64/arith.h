//
// Created by Corrado Mio on 08/03/2024.
//
#include <cmath>

#ifndef STDX_FLOAT64_ARITH_H
#define STDX_FLOAT64_ARITH_H

namespace stdx::arith {

    inline double neg(double x)           { return -x;    }
    inline double  sq(double x)           { return x * x; }
    inline double sum(double x, double y) { return x + y; }
    inline double sub(double x, double y) { return x - y; }
    inline double mul(double x, double y) { return x * y; }
    inline double div(double x, double y) { return x / y; }
    inline double eps(double x, int n=3) { return x*std::pow(2.,-16+n); }
}

#endif //STDX_FLOAT64_ARITH_H
