//
// Created by Corrado Mio on 26/07/2023.
//

#ifndef CHECK_HALF_IEEE754STD_H
#define CHECK_HALF_IEEE754STD_H

#include <cmath>

namespace ieee754 {

    // math.h
    template<typename T> bool isfinite(T x);
    template<typename T> bool isinf(T x);
    template<typename T> bool isnormal(T x);
    template<typename T> bool isnan(T x);
    template<typename T> bool signbit(T x);
    template<typename T> int  fpclassify(T x);

    // math.h
    template<typename T> bool isgreater(T x, T y);
    template<typename T> bool isgreaterequal (T x, T y);
    template<typename T> bool isless(T x, T y);
    template<typename T> bool islessequal(T x, T y);
    template<typename T> bool islessgreater(T x, T y);
    template<typename T> bool isunordered(T x, T y);

    // extra
    template<typename T> bool isequal(T x, T y)          { return x == y; }
    template<typename T> bool isequal(T x, T y, T eps)   { return abs(x, y) <= eps; }
    template<typename T> bool isless(T x, T y, T eps)    { return x < (y-eps); }
    template<typename T> bool isgreater(T x, T y, T eps) { return isless(y, x); }

    // float
    // extern template bool isfinite<float>(float x);
    // extern template bool isinf<float>(float x);
    // extern template bool isnormal<float>(float x);
    // extern template bool isnan<float>(float x);
    // extern template bool signbit<float>(float x);
    // extern template int  fpclassify<float>(float x);

    // extern template bool isgreater(float x, float y);
    // extern template bool isgreaterequal (float x, float y);
    // extern template bool isless(float x, float y);
    // extern template bool islessequal(float x, float y);
    // extern template bool islessgreater(float x, float y);
    // extern template bool isunordered(float x, float y);

    // double
    // extern template bool isfinite<double>(double x);
    // extern template bool isinf<double>(double x);
    // extern template bool isnormal<double>(double x);
    // extern template bool isnan<double>(double x);
    // extern template bool signbit<double>(double x);
    // extern template int  fpclassify<double>(double x);
    //
    // extern template bool isgreater(double x, double y);
    // extern template bool isgreaterequal (double x, double y);
    // extern template bool isless(double x, double y);
    // extern template bool islessequal(double x, double y);
    // extern template bool islessgreater(double x, double y);
    // extern template bool isunordered(double x, double y);

    // math.h
    // INFINITY
    // NAN
    // HUGE_VAL
    // HUGE_VALF
    // HUGE_VALL
    //
    // MATH_ERRNO
    // MATH_ERREXCEPT
    //
    // fpclassify
    // FP_NAN           0x0100
    // FP_NORMAL        0x0400
    // FP_ZERO          0x4000
    // FP_SUBNORMAL     (FP_NORMAL | FP_ZERO)
    // FP_INFINITE      (FP_NAN | FP_NORMAL)
    //
    // FP_ILOGB0        special for ilogb
    // FP_ILOGBNAN      special for ilogb


    // math.h
    template<> inline bool isfinite<float>(float x) { return std::isfinite(x); }
    template<> inline bool isinf(float x) { return std::isinf(x); }
    template<> inline bool isnormal(float x) { return std::isnormal(x); }
    template<> inline bool isnan(float x) { return std::isnan(x); }
    template<> inline bool signbit(float x) { return std::signbit(x); }
    template<> inline int fpclassify(float x) { return std::fpclassify(x); }

    // math.h
    template<> constexpr bool isgreater(float x, float y) { return std::isgreater(x, y); }
    template<> constexpr bool isgreaterequal (float x, float y) { return std::isgreaterequal(x, y); }
    template<> constexpr bool isless(float x, float y) { return std::isless(x, y); }
    template<> constexpr bool islessequal(float x, float y) { return std::islessequal(x, y); }
    template<> constexpr bool islessgreater(float x, float y) { return std::islessgreater(x, y); }
    template<> constexpr bool isunordered(float x, float y) { return std::isunordered(x, y); }

};

#endif //CHECK_HALF_IEEE754STD_H
