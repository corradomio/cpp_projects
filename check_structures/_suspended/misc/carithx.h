//
// Created by Corrado Mio on 29/02/2024.
//

#ifndef CARITHX_H
#define CARITHX_H

namespace stdx::math {

    template<typename T>    T neg(T x)      { return -x; }
    template<typename T>    T  sq(T x)      { return x*x; }
    template<typename T>    T sum(T x, T y) { return x+y; }
    template<typename T>    T sub(T x, T y) { return x-y; }
    template<typename T>    T mul(T x, T y) { return x*y; }
    template<typename T>    T div(T x, T y) { return x/y; }
}

#endif //CARITHX_H
