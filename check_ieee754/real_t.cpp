//
// Created by Corrado Mio on 28/07/2023.
//
/*
    typedef real_t<10, 53, uint64_t> float64_t;
    typedef real_t<8, 23,  uint32_t> float32_t;
    typedef real_t<8, 7,   uint16_t> bfloat16_t;
    typedef real_t<5, 10,  uint16_t> float16_t;
    typedef real_t<4, 3,   uint8_t>  bfloat8_t;
    typedef real_t<5, 2,   uint8_t>  float8_t;
 */

#include "real_t.h"

#define self (*this)

using namespace ieee754;

template<> real_t<10, 53, uint64_t>::real_t(int i) {
    ieee754_t t;
    t.d = i;
    self.s = t.f64.s;
    self.e = t.f64.e;
    self.m = t.f64.m;
}