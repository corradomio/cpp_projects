//
// Created by Corrado Mio on 28/07/2023.
//

#ifndef CHEC_IEEE754_REAL_T_H
#define CHEC_IEEE754_REAL_T_H

#include <cstdint>

namespace ieee754 {

    template<int E, int M, typename T>
    union real_t {
        struct {uint8_t s:1; uint16_t e:E; T m:M; };
        T data;
        static const T EBIAS = ((1<<(E-1))-1);

        real_t() { };
        real_t(const real_t& r): data(r.data) { }

        real_t& operator =(const real_t& r) { data=r.data; return *this; }
    };

    typedef real_t<10, 53, uint64_t> float64_t;
    typedef real_t<8, 23,  uint32_t> float32_t;
    typedef real_t<8, 7,   uint16_t> bfloat16_t;
    typedef real_t<5, 10,  uint16_t> float16_t;
    typedef real_t<4, 3,   uint8_t>  bfloat8_t;
    typedef real_t<5, 2,   uint8_t>  float8_t;

    /*
     * Conversions float/double <--> integer/{s, e, m}
     */

    template<typename T, typename U> T from_bits(bool s, uint16_t e, U m);
    template<typename T, typename U> T from_bits(U u);
    template<typename T, typename U> U to_bits(T u);

    // float
    template<> inline float from_bits<float>(bool s, uint16_t e, uint32_t m) {
        union {
            struct {uint32_t m:23; uint32_t e:8; uint32_t s:1; };
            float f;
        } u;
        u.s = s;
        u.e = e;
        u.m = m;
        return u.f;
    }

    template<> inline float from_bits<float>(uint32_t l) {
        union {
            uint32_t l;
            float f;
        } u;
        u.l = l;
        return u.f;
    }

    template<> inline unsigned long to_bits<float>(float f) {
        union {
            uint32_t l;
            float f;
        } u;
        u.f = f;
        return u.l;
    }

    // double
    template<> inline double from_bits<double>(bool s, uint16_t e, uint64_t m) {
        union {
            struct {uint64_t m:52; uint32_t e:11; uint32_t s:1; };
            double f;
        } u;
        u.s = s;
        u.e = e;
        u.m = m;
        return u.f;
    }

    template<> inline double from_bits<double>(uint64_t l) {
        union {
            uint64_t l;
            double f;
        } u;
        u.l = l;
        return u.f;
    }

    template<> inline unsigned long long to_bits<double>(double f) {
        union {
            unsigned long l;
            double f;
        } u;
        u.f = f;
        return u.l;
    }

} // ieee754

#endif //CHEC_IEEE754_REAL_T_H
