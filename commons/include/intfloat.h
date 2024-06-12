//
// Created by Corrado Mio on 02/06/2024.
//
// Alternative names for standard types

#ifndef STDX_INTFLOAT_H
#define STDX_INTFLOAT_H

#include <stdint.h>
//  size:   8 | 16 | 32 | 64 | 128?
//
//  int<size>_t         uint<size>_t
//  int_least<size>_t   uint_least<size>_t
//  int_fast<size>_t    uint_fast<size>_t
//  intmax_t            uintmax_t
//
//  INT<size>_MIN           INT<size>_MAX           UINT<size>_MAX
//  INT_LEAST<size>_MIN     INT_LEAST<size>_MAX     UINT_LEAST<size>_MAX
//  INT_FAST<size>_MIN      INT_FAST<size>_MAX      UINT_FAST<size>_MAX
//  INTPTR_MIN              INTPTR_MAX              UINTPTR_MAX
//  PTRDIFF_MIN             PTRDIFF_MAX
//  SIG_ATOMIC_MIN          SIG_ATOMIC_MAX
//  WCHAR_MIN               WCHAR_MAX
//  WINT_MIN                WINT_MAX


typedef _Float16 __float16;             // supported
typedef __bf16   __bfloat16;            // supported ?
typedef float    __float32;             // compatibility
typedef double   __float64;             // compatibility
typedef          __int128 int128_t;     // compatibility
typedef unsigned __int128 uint128_t;    // compatibility

typedef _Float16  f16;
typedef __bf16    bf16;
typedef float     f32;
typedef double    f64;

typedef int8_t    i8;
typedef int16_t   i16;
typedef int32_t   i32;
typedef int64_t   i64;
typedef int128_t  i128;

typedef uint8_t   u8;
typedef uint16_t  u16;
typedef uint32_t  u32;
typedef uint64_t  u64;
typedef uint128_t u128;

#endif //STDX_INTFLOAT_H
