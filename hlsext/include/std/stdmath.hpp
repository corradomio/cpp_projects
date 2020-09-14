//
// Created by Corrado Mio on 11/05/2015.
//

#ifndef TEST_STDMATH_HPP
#define TEST_STDMATH_HPP

#include <cmath>

namespace std {

    template<typename T> T round(T x);
    template<> double round<double>(double x) { return ::round(x);  }
    template<> float  round< float>(float  x) { return ::roundf(x); }

    template<typename T> T sqrt(T x);
    template<> double sqrt<double>(double x) { return ::sqrt(x);  }
    template<> float  sqrt< float>(float  x) { return ::sqrtf(x); }

    template<typename T> T sq(T x) { return x*x; }
    template<typename T> T dist(T x, T y, T z) { return sqrt(sq(x) + sq(y) + sq(z)); }

}

#endif //TEST_STDMATH_HPP
