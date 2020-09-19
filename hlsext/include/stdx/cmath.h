//
// Created by Corrado Mio on 19/09/2020.
//

#ifndef CHECK_HLSEXT_CMATH_H
#define CHECK_HLSEXT_CMATH_H

#include<cmath>

#ifndef _GLIBCXX_USE_C99_MATH_TR1

namespace std {

    constexpr float fmax(float __x, float __y) { return __builtin_fmaxf(__x, __y); }
    constexpr long double fmax(long double __x, long double __y) { return __builtin_fmaxl(__x, __y); }

    template<typename _Tp, typename _Up>
    constexpr typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    fmax(_Tp __x, _Up __y)
    {
        typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
        return fmax(__type(__x), __type(__y));
    }


    constexpr float fmin(float __x, float __y) { return __builtin_fminf(__x, __y); }
    constexpr long double fmin(long double __x, long double __y) { return __builtin_fminl(__x, __y); }

    template<typename _Tp, typename _Up>
    constexpr typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    fmin(_Tp __x, _Up __y)
    {
        typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
        return fmin(__type(__x), __type(__y));
    }


    constexpr float tgamma(float __x) { return __builtin_tgammaf(__x); }
    constexpr long double tgamma(long double __x) { return __builtin_tgammal(__x); }

    template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type
    tgamma(_Tp __x)
    { return __builtin_tgamma(__x); }


    constexpr float lgamma(float __x) { return __builtin_lgammaf(__x); }
    constexpr long double lgamma(long double __x) { return __builtin_lgammal(__x); }

    template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type
    lgamma(_Tp __x)
    { return __builtin_lgamma(__x); }


    constexpr float erf(float __x) { return __builtin_erff(__x); }
    constexpr long double erf(long double __x) { return __builtin_erfl(__x); }

    template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type
    erf(_Tp __x)
    { return __builtin_erf(__x); }


    constexpr float erfc(float __x) { return __builtin_erfcf(__x); }
    constexpr long double erfc(long double __x) { return __builtin_erfcl(__x); }

    template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type
    erfc(_Tp __x)
    { return __builtin_erfc(__x); }


    constexpr float atanh(float __x) { return __builtin_atanhf(__x); }
    constexpr long double atanh(long double __x) { return __builtin_atanhl(__x); }

    template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type
    atanh(_Tp __x)
    { return __builtin_atanh(__x); }


    /// Additional overloads.
    constexpr float acosh(float __x) { return __builtin_acoshf(__x); }
    constexpr long double acosh(long double __x) { return __builtin_acoshl(__x); }

    template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type
    acosh(_Tp __x)
    { return __builtin_acosh(__x); }


    constexpr float asinh(float __x) { return __builtin_asinhf(__x); }
    constexpr long double asinh(long double __x) { return __builtin_asinhl(__x); }

    template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type
    asinh(_Tp __x)
    { return __builtin_asinh(__x); }


    constexpr float hypot(float __x, float __y) { return __builtin_hypotf(__x, __y); }
    constexpr long double hypot(long double __x, long double __y) { return __builtin_hypotl(__x, __y); }

    template<typename _Tp, typename _Up>
    constexpr typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    hypot(_Tp __x, _Up __y)
    {
        typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
        return hypot(__type(__x), __type(__y));
    }


    constexpr float cbrt(float __x) { return __builtin_cbrtf(__x); }
    constexpr long double cbrt(long double __x) { return __builtin_cbrtl(__x); }

    template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type
    cbrt(_Tp __x)
    { return __builtin_cbrt(__x); }


    // DR 568.
    constexpr float log2(float __x) { return __builtin_log2f(__x); }
    constexpr long double log2(long double __x) { return __builtin_log2l(__x); }

    template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type
    log2(_Tp __x)
    { return __builtin_log2(__x); }


    constexpr float log1p(float __x) { return __builtin_log1pf(__x); }
    constexpr long double log1p(long double __x) { return __builtin_log1pl(__x); }

    template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type
    log1p(_Tp __x)
    { return __builtin_log1p(__x); }


    constexpr float exp2(float __x) { return __builtin_exp2f(__x); }
    constexpr long double exp2(long double __x) { return __builtin_exp2l(__x); }

    template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type
    exp2(_Tp __x)
    { return __builtin_exp2(__x); }


    constexpr float expm1(float __x) { return __builtin_expm1f(__x); }
    constexpr long double expm1(long double __x) { return __builtin_expm1l(__x); }

    template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type
    expm1(_Tp __x)
    { return __builtin_expm1(__x); }


    constexpr float fdim(float __x, float __y) { return __builtin_fdimf(__x, __y); }
    constexpr long double fdim(long double __x, long double __y) { return __builtin_fdiml(__x, __y); }

    template<typename _Tp, typename _Up>
    constexpr typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    fdim(_Tp __x, _Up __y)
    {
        typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
        return fdim(__type(__x), __type(__y));
    }


    inline float remquo(float __x, float __y, int* __pquo) { return __builtin_remquof(__x, __y, __pquo); }
    inline long double remquo(long double __x, long double __y, int* __pquo) { return __builtin_remquol(__x, __y, __pquo); }

    template<typename _Tp, typename _Up>
    inline typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    remquo(_Tp __x, _Up __y, int* __pquo)
    {
        typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
        return remquo(__type(__x), __type(__y), __pquo);
    }


    constexpr float remainder(float __x, float __y) { return __builtin_remainderf(__x, __y); }
    constexpr long double remainder(long double __x, long double __y) { return __builtin_remainderl(__x, __y); }

    template<typename _Tp, typename _Up>
    constexpr typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    remainder(_Tp __x, _Up __y)
    {
        typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
        return remainder(__type(__x), __type(__y));
    }

};

#endif

namespace stdx {
namespace math {

    const float  maxfloat  = 3.40282347e+38F;
    const double maxdouble = 0;

    const double e      = 2.7182818284590452354;
    const double log2e  = 1.4426950408889634074;    //
    const double log10e = 0.43429448190325182765;
    const double ln2    = 0.693147180559945309417;
    const double ln10   = 2.30258509299404568402;

    const double pi     = 3.14159265358979323846;   // PI
    const double pi2    = 1.57079632679489661923;   // PI/2
    const double pi4    = 0.78539816339744830962;   // PI/4
    const double invpi  = 0.31830988618379067154;   // 1/ PI

    const double twopi  = (2.0*pi);
    const double sqrtpi = 1.77245385090551602792981;
    const double sqrt3  = 1.73205080756887719000;

    const double invln10 = 0.43429448190325182765;
    const double intln2  = 1.44269504088896338700;


    template<typename T> T round(T x);
    template<> double round<double>(double x) { return ::round(x);  }
    template<> float  round< float>(float  x) { return ::roundf(x); }

    template<typename T> T sqrt(T x);
    template<> double sqrt<double>(double x) { return ::sqrt(x);  }
    template<> float  sqrt< float>(float  x) { return ::sqrtf(x); }


    template<typename T> T sq(T x) { return x*x; }
    template<typename T> T dist(T x, T y, T z) { return sqrt(sq(x) + sq(y) + sq(z)); }

}};

#endif //CHECK_HLSEXT_CMATH_H
