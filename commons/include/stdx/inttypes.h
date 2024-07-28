//
// Created by Corrado Mio (Local) on 23/04/2021.
//

#ifndef STDX_INTTYPES_H
#define STDX_INTTYPES_H

#include <cstdint>

namespace stdx {
    typedef bool                bool_t;

    typedef signed char         int8_t,   i8;
    typedef signed short        int16_t,  i16;
    typedef signed long         int32_t,  i32;
    typedef signed long long    int64_t,  i64;
    typedef signed __int128     int128_t, i128;

    typedef unsigned char       uint8_t,  u8, byte_t;
    typedef unsigned short      uint16_t, u16, word_t;
    typedef unsigned long       uint32_t, u32, dword_t;
    typedef unsigned long long  uint64_t, u64, qword_t;
    typedef unsigned __int128   uint128_t, u128;
}

#endif // STDX_INTTYPES_H
