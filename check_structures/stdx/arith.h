//
// Created by Corrado Mio on 08/03/2024.
//
#include <cmath>

#ifndef STDX_FLOAT64_ARITH_H
#define STDX_FLOAT64_ARITH_H

namespace stdx {

    double neg(double x)           { return -x;    }
    double  sq(double x)           { return x * x; }
    double sum(double x, double y) { return x + y; }
    double sub(double x, double y) { return x - y; }
    double mul(double x, double y) { return x * y; }
    double div(double x, double y) { return x / y; }

    double eps(double x, int n=3) { return x*std::pow(2.,-16+n); }
}

#endif //STDX_FLOAT64_ARITH_H
