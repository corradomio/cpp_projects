//
// Created by Corrado Mio on 28/07/2023.
//

#ifndef self
#define self (*this)
#endif

#ifndef elif
#define elif else if
#endif


#ifndef IEEE754_REAL_T_H
#define IEEE754_REAL_T_H

#include <cmath>
#include <cstdint>
#include <intrin.h>

/*
 *
 * Formats IEEE 754
 * ----------------
 *
 *      +-+---+--------+
 *      |s|e  |m       |
 *      +-+---+--------+
 *
 *
 *      s | e  | m  | desc
 *      --+----+----+------------
 *        |  0 |  0 | zero
 *        |  0 | ++ | subnormal     v = (-1)^s 0.m * 2^(e-bias)
 *        | >0 | ++ | normalized    v = (-1)^s 1.m * 2^(e-bias)     '1' implicit
 *        | 11 |  0 | infinity      v = (-1)^s infinity
 *        | 11 | ++ | NaN           v = Not a Number, s&m specify which type
 *      --+----+----+------------
 *
 * Checks
 * ------
 *
 *      fpclassify -> FP_INFINITE, FP_NAN, FP_NORMAL, FP_SUBNORMAL, FP_ZERO
 *          isnan
 *          isfinite
 *          isinf
 *          isnormal
 *          isunordered
 *
 * Standard formats
 * ----------------
 *
 *  format:  E<e.len>M<m.len>
 *
 *  float128_t  [15,112]    https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format
 *  float80_t   [15,64]     https://en.wikipedia.org/wiki/Extended_precision
 *  double      [11,52]     https://en.wikipedia.org/wiki/IEEE_754-1985
 *  float       [ 8,23]     https://en.wikipedia.org/wiki/IEEE_754-1985
 *  half        [ 5,10]     https://en.wikipedia.org/wiki/Half-precision_floating-point_format
 *  bloat16     [ 8, 7]     https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
 *  float8      [ 4, 3]     https://en.wikipedia.org/wiki/Minifloat
 *  mfloat8     [ 3, 4]     https://people.cs.umass.edu/~verts/cmpsci145/8-Bit_Floating_Point.pdf
 *  bfloat8     [ 5, 2]     https://towardsdatascience.com/16-8-and-4-bit-floating-point-formats-how-does-it-work-d157a31ef2ef
 *
 *  tfloat      [ 8,10]     https://en.wikipedia.org/wiki/Bfloat16_floating-point_format    (19 bit)
 *  amdfp24     [ 7,16]     https://en.wikipedia.org/wiki/Bfloat16_floating-point_format    (24 bits)
 *  pxr24       [ 8,15]     https://en.wikipedia.org/wiki/Bfloat16_floating-point_format    (24 bits)
 *
 *  fp8: E3M4 E5M2      https://arxiv.org/pdf/2209.05433.pdf
 *  ms-fp8, ms-fp9
 *
 *
 * Implementation problems
 * -----------------------
 *
 * GNU C++ compiler
 *
 *      __float128
 *      __float80  _Float64x
 *      _Float64 _Float32 _Float32x _Float16
 *      __fp16
 *
 *      float80_t   it is necessary to use an int80_t to specify the bit structure
 *                  the alternative implementation, requiring a deep implementation
 *                  is to use byte[10], or to implement 'int80_t' (or '__int80')
 *                  The current implementation, float80 is represented with 128 bit
 *
 *      float128_t  in theory it is possible to define it because is available '__int180'
 *                  and '__float128'
 *                  but it is difficult touse it because there are strange behaviour.
 *                  It is not clear IF it is a problem of the debugger or of the compiler.
 *                  Or in some other part of the implementation. Hoeever, to print the
 *                  min/max/epsilon values from 'numerical_limit<long double>' works
 *                  correctly. It is not clear IF 'long double' is at 80 bits or 128 bits.
 *                  But we have that 'sizeof(long double) == 16' (128 bit!!)
 *
 *      minifloat   it is not useful to implement minifloat because the minimum allocation
 *                  space is 1 byte.
 *
 *      zerofloat   at the moment it is not supported floating in format E<len>M0 or E0M<len>
 *
 *
 * Other notes
 * -----------
 *
 *      'long double'  is __float128 ??
 *
 *      __float128      ok
 *      __float80       ok  same size than __float128
 *      _Float64x       ok  same size than __float128
 *      __fp16          no
 *      __float16       ok
 *      _Float16        ok  -> defined __float16
 *      __bf16          ok  -> defined __bfloat16
 *
 *      __int8 __int16 __int32 __int64 __int128        ok
 *
 */

//
// normalize some definitions
//

typedef _Float16 __float16;             // supported
typedef __bf16   __bfloat16;            // supported ?
typedef float    __float32;             // compatibility
typedef double   __float64;             // compatibility
typedef          __int128 int128_t;     // compatibility
typedef unsigned __int128 uint128_t;    // compatibility


namespace ieee754 {

    // convert each format in each other
    template<typename Target, typename Source>
    Target fpconvert(const Source& s);

    // ----------------------------------------------------------------------
    // real_t<E,M>
    // ----------------------------------------------------------------------
    //
    //  +-----+----------+----------------------------------+
    //  | s:1 | e:E      | m:M                              |
    //  +-----+----------+----------------------------------+
    //
    //

    template<int E, int M, typename T>
    union real_t {

        // WARNING: the field ORDER MUST BE SPECIFIED in OPPOSITE WAY: (m, e, s) !!!
        struct { T m:M; T e:E; T s:1; };
        T data;

        // ------------------------------------------------------------------
        // properties

        typedef T field_type;

        static const int EMAX  = ((1<<E)-1);
        static const int EBIAS = (EMAX>>1);
        static const int ELEN  = E;
        static const int MLEN  = M;

        void assign(const real_t& that) {
            self.s = that.s;
            self.e = that.e;
            self.m = that.m;
        }

        // ------------------------------------------------------------------
        // constructor

        constexpr real_t() { };
        constexpr real_t(const real_t& that) {  assign(that); }

        // bit fields constructor
        constexpr real_t(T s, T e, T m) {
            self.s = s;
            self.e = e;
            self.m = m;
        }

        // ------------------------------------------------------------------
        // assignment

        real_t& operator =(const real_t& that) {
            assign(that);
            return self;
        }

        // ------------------------------------------------------------------
        // conversions
        // ------------------------------------------------------------------
        // Warn: it is not possible to define the conversions here because they
        // depends on float_u/double_u data structures
        // Warn: it is not possible to define the templated versions here because
        // this is in conflict with the specialized implementations based on
        // float/double/long double

        // constructors
        template<typename Source> real_t(const Source &s);
        [[nodiscard]] explicit real_t(float flt);
        [[nodiscard]] explicit real_t(double dbl);
        // [[nodiscard]] explicit real_t(long double dbl);

        // assignments
        template<typename Source> real_t& operator =(const Source& s);
        real_t& operator =(float flt);
        real_t& operator =(double dbl);
        // real_t& operator =(long double dbl);

        // type operators
        template<typename Target> operator Target() const;
        [[nodiscard]] operator float() const;
        [[nodiscard]] operator double() const;
        // [[nodiscard]] operator long double() const;

    };

    // ----------------------------------------------------------------------
    // useful alternative definitions
    // ----------------------------------------------------------------------

    typedef real_t<15,112, uint128_t> float128_t;
    typedef real_t<11,52,uint64_t> float64_t;
    typedef real_t< 8,23,uint32_t> float32_t;
    typedef real_t< 8, 7,uint16_t> bfloat16_t;
    typedef real_t< 5,10,uint16_t> float16_t;
    typedef real_t< 5, 2,uint8_t > bfloat8_t;
    typedef real_t< 4, 3,uint8_t > float8_t;

    // ----------------------------------------------------------------------
    // union types used in conversions
    // ----------------------------------------------------------------------

    union float64_u {
        float64_t f64;
        double    dbl;
        uint64_t  u64;
        float64_u(){}
        float64_u(double d): dbl(d){}
    };

    union float32_u {
        float32_t f32;
        float     flt;
        uint32_t  u32;
        float32_u(){}
        float32_u(float f): flt(f){}
    };

    union float16_u {
        float16_t f16;
        __float16 flt;
        uint16_t  u32;
        float16_u(){}
        float16_u(__float16 f): flt(f){}
    };


    // ----------------------------------------------------------------------
    // fpclassify
    // ----------------------------------------------------------------------

    template<int E, int M, typename T>
    inline int fpclassify(real_t<E,M,T> f) {
        if (f.e == 0)
            return (f.m == 0) ? FP_ZERO : FP_SUBNORMAL;
        if (f.e == f.EMAX)
            return (f.m == 0) ? FP_INFINITE : FP_NAN;
        else
            return FP_NORMAL;
    }

    // ----------------------------------------------------------------------
    // conversions implementation
    // special conversions from/to float/double/long double

    template<typename Target, typename Source>
    Target fpconvert(const Source& s) {
        Target t;
        _fpconvert(t, s);
        return t;
    }

    template<typename Target, typename Source>
    void _fpconvert(Target & t, const Source & s) {

        typedef typename Source::field_type source_type;
        typedef typename Target::field_type target_type;

        if (s.e == 0 && s.m == 0) {
            // zero
            t.s = s.s;
            t.e = s.e;
            t.m = s.m;
            return;
        }
        elif (s.e == s.EMAX) {
            // infinity & nan
            t.s = s.s;
            t.e = t.EMAX;
            t.m = s.m;
            return;
        }

        int e = (s.e - s.EBIAS + t.EBIAS);
        int dm = Target::MLEN - Source::MLEN;
        int de = Target::ELEN - Source::ELEN;

        // copy sign & exponent
        // align the mantissa
        {
            t.s = s.s;
            t.e = e;

            // |t.m| >= |s.m|
            if (dm >= 0)
                t.m = target_type(s.m) << (+dm);
            // |t.m| < |s.m|
            else
                t.m = source_type(s.m) >> (-dm);
        }

        // s == 0 {s.e == 0, s.m == 0} already processed

        // |t.e| > |s.e|
        if (de > 0) {
            // e = s.e - s.EBIAS + t.EBIAS
            // s.EBIAS < t.EBIAS
            // e is ALWAYS positive!

            // normal remains normal
            if (s.e > 0) return;

            // s.e == 0 AND s.m != 0 because the case
            // s.m == 0 is already processed

            // subnormal can became normal
            int hibit = _bit_scan_reverse(t.m);
            // available zeros in front of the highest bit
            int zm = Target::MLEN - hibit - 1;

            if (e <= zm) {
                // there are ENOUGH zero bits
                // the subnormal remains subnormal
                t.e = 0;
                t.m <<= e;
            }
            else {
                // the are NOT ENOUGH zero bits
                // the subnormal became normal

                t.e = e - zm;
                t.m <<= (zm+1);
            }
        }
        // |t.e| < |s.e|
        elif ( de < 0) {
            // subnormal remain subnormal
            if (s.e == 0) {
                // because e = s.e - s.EBIAS + t.EBIAS
                // and s.e == 0 AND s.EBIAS > t.EBIAS
                // e is NEGATIVE
                int f = -e;

                // int hibit = _bit_scan_reverse(t.m);
                // if (f > hibit) {
                //     // shift right outside the right side
                //     t.e = 0;
                //     t.m = 0;
                // }
                // else {
                //     // shift right inside the mantissa
                //     t.e = 0;
                //     t.m >>= f;
                // }

                if (f >= Target::MLEN) {
                    // shift right outside the right side
                    t.e = 0;
                    t.m = 0;
                }
                else {
                    // shift right inside the mantissa
                    t.e = 0;
                    t.m >>= f;
                }

                // WARNING: it doesn't work. It is used the previous code
                // t.e = 0;
                // t.m >>= f;
            }
            // normal became subnormal
            elif (e < 0) {
                typename Target::field_type ONE = 1;
                int f = -e;

                if (f > Target::MLEN) {
                    // shift the mantissa right outside the right side
                    t.e = 0;
                    t.m = 0;
                }
                // elif (f == Target::MLEN) {
                //     // shift the mantissa right outside the right side
                //     // BUT the implicit '1' in the last positions
                //     t.e = 0;
                //     t.m = 1;
                // }
                else {
                    t.e = 0;
                    t.m = (t.m | (ONE << Target::MLEN)) >> (f+1);
                };

                // t.e = 0;
                // t.m = (t.m | (ONE << Target::MLEN)) >> (f+1);
            }
        }
        else {
            // |t.e| == |s.e|
            // none to do
        }
    }

    // ----------------------------------------------------------------------
    // float/double -> Target

    template<typename Target>
    Target fpconvert(float flt) {
        float32_u u(flt);
        Target t;
        _fpconvert<Target>(t, u.f32);
        return t;
    }

    template<typename Target>
    Target fpconvert(double dbl) {
        float64_u u(dbl);
        Target t;
        _fpconvert<Target>(t, u.f64);
        return t;
    }


    // ----------------------------------------------------------------------
    // Source -> float/double

    template<typename Source>
    float fpconvert(const Source& s) {
        float32_u u;
        _fpconvert(u.f32, s);
        return u.flt;
    }

    template<typename Source>
    double fpconvert(const Source& s) {
        float64_u u;
        _fpconvert(u.f64, s);
        return u.dbl;
    }


    // ----------------------------------------------------------------------
    // float|double|conversion constructors
    // real_t(real_t)
    // real_t(float)
    // real_t(double)

    template<int E, int M, typename T>
    template<typename Source>
    real_t<E,M,T>::real_t(const Source &s) {
        _fpconvert(self, s);
    }

    template<int E, int M, typename T>
    real_t<E,M,T>::real_t(float flt) {
        float32_u u(flt);
        _fpconvert(self, u.f32);
    }

    template<int E, int M, typename T>
    real_t<E,M,T>::real_t(double dbl) {
        float64_u u(dbl);
        _fpconvert(self, u.f64);
    }

    // ----------------------------------------------------------------------
    // float|double|conversion operators
    // (real_t)(...)
    // (float)(...)
    // (double)(...)

    template<int E, int M, typename T>
    template<typename Target>
    real_t<E,M,T>::operator Target() const {
        Target t;
        _fpconvert(t,self);
        return t;
    }

    template<int E, int M, typename T>
    real_t<E,M,T>::operator float() const {
        float32_u u;
        _fpconvert(u.f32,self);
        return u.flt;
    }

    template<int E, int M, typename T>
    real_t<E,M,T>::operator double() const {
        float64_u u;
        _fpconvert(u.f64,self);
        return u.dbl;
    }


    // ----------------------------------------------------------------------
    // float|double|conversion assignment
    // real_t = real_t
    // real_t = float
    // real_t = double

    template<int E, int M, typename T>
    template<typename Source>
    real_t<E,M,T>& real_t<E,M,T>::operator =(const Source& s) {
        _fpconvert(self, s);
        return self;
    }

    template<int E, int M, typename T>
    real_t<E,M,T>& real_t<E,M,T>::operator =(float flt) {
        float32_u u(flt);
        _fpconvert(self, u.f32);
        return self;
    }

    template<int E, int M, typename T>
    real_t<E,M,T>& real_t<E,M,T>::operator =(double dbl) {
        float64_u u(dbl);
        _fpconvert(self, u.f64);
        return self;
    }

    // end
    // ----------------------------------------------------------------------

    template<int TE, int TM, typename TT, int SE, int SM, typename ST>
    bool cmp_e(const real_t<TE,TM,TT>& t, const real_t<SE,SM,ST>& s) {
        return (t.e - t.EBIAS) == (s.e - s.EBIAS);
    }

    template<int TE, int TM, typename TT, int SE, int SM, typename ST>
    bool cmp_m(const real_t<TE,TM,TT>& t, const real_t<SE,SM,ST>& s) {
        if (t.MLEN < s.MLEN) {
            TT sm = (s.m >> (s.MLEN - t.MLEN));
            return t.m == sm;
        }
        elif (t.MLEN > s.MLEN) {
            ST tm = (t.m >> (t.MLEN - s.MLEN));
            return tm == s.m;
        }
        else {
            return t.m == s.m;
        }
    }

    template<int TE, int TM, typename TT, int SE, int SM, typename ST>
    bool operator ==(const real_t<TE,TM,TT>& t, const real_t<SE,SM,ST>& s) {
        return t.s == s.s
            && cmp_e(t, s)
            && cmp_m(t, s);
    }

    template<int E, int M, typename T>
    bool operator ==(const real_t<E,M,T>& t, float flt) {
        float32_u u(flt);
        return t == u.f32;
    }

    template<int E, int M, typename T>
    bool operator ==(const real_t<E,M,T>& t, double dbl) {
        float64_u u(dbl);
        return t == u.f64;
    }

    template<int E, int M, typename T>
    bool operator ==(float flt, const real_t<E,M,T>& t) {
        float32_u u(flt);
        return t == u.f32;
    }

    template<int E, int M, typename T>
    bool operator ==(double dbl, const real_t<E,M,T>& t) {
        float64_u u(dbl);
        return t == u.f64;
    }


    // end
    // ----------------------------------------------------------------------


} // ieee754

#endif //IEEE754_REAL_T_H
