//
// Created by Corrado Mio (Local) on 23/04/2021.
//

#ifndef STDX_INTTYPES_H
#define STDX_INTTYPES_H

#include <cstdint>

namespace stdx {
    typedef bool     bool_t;

    typedef signed char  int8_t;
    typedef signed short int16_t;
    typedef signed long  int32_t;
    typedef signed long long int64_t;

    typedef unsigned char  uint8_t;
    typedef unsigned short uint16_t;
    typedef unsigned long  uint32_t;
    typedef unsigned long long uint64_t;

    typedef float  float32_t;
    typedef double float64_t;

    typedef unsigned char  byte_t;
    typedef unsigned short word_t;
    typedef unsigned long  dword_t;
    typedef unsigned long long qword_t;

    typedef float  real32_t;
    typedef double real64_t;
}

#endif // STDX_INTTYPES_H
