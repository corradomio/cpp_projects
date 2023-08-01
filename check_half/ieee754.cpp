//
// Created by Corrado Mio on 25/07/2023.
//
#include <cmath>
#include "ieee754.h"

using namespace ieee754;

// float32 float32::add(const float32& f1, const float32& f2) {
//     uint32_t s, e, de, m1, m2, m;
//     float32 r;
//
//     s = f1.s;
//
//     // e1 < e2
//     if (f1.e < f2.e) {
//         e = f2.e;
//         de = e - f1.e;
//         m1 = (f1.m | M_MAX) >> de;
//         m2 = (f2.m | M_MAX);
//     }
//     // e1 > e2
//     else if (f1.e > f2.e) {
//         e = f1.e;
//         de = e - f2.e;
//         m1 = (f1.m | M_MAX);
//         m2 = (f2.m | M_MAX) >> de;
//     }
//     // e1 == e2
//     else {
//         e = f1.e;
//         m1 = (f1.m | M_MAX);
//         m2 = (f2.m | M_MAX);
//     }
//
//     m = m1+m2;
//     while (m >= M_TOP) {
//         e += 1;
//         m >>= 1;
//     }
//     m ^= M_MAX;
//
//     r.s = s;
//     r.e = e;
//     r.m = m;
//     return r;
// }

float32 ieee754::add(const float32& f1, const float32& f2) {
    uint32_t s, e, de, m1, m2, m;
    float32 r;

    s = f1.s;

    // e1 < e2
    if (f1.e < f2.e) {
        e = f2.e;
        de = e - f1.e;
        m1 = (f1.m | M_MAX) >> de;
        m2 = f2.m;
        m = m1+m2;
    }
    // e1 > e2
    else if (f1.e > f2.e) {
        e = f1.e;
        de = e - f2.e;
        m1 = f1.m;
        m2 = (f2.m | M_MAX) >> de;
        m = m1+m2;
    }
    // e1 == e2
    else {
        e = f1.e + 1;
        m1 = f1.m;
        m2 = f2.m;
        m = m1+m2;
        m >>= 1;
    }

    if (m >= M_MAX) {
        e += 1;
        m >>= 1;
    }

    r.s = s;
    r.e = e;
    r.m = m;
    return r;
}

float32 ieee754::sub(const float32& f1, const float32& f2) {
    uint32_t s, e, de, m1, m2, m;
    float32 r;

    // e1 < e2
    if (f1.e < f2.e) {
        e = f2.e;
        de = e - f1.e;
        m1 = (f1.m | M_MAX) >> de;
        m2 = (f2.m | M_MAX);
    }
    // e1 > e2
    else if (f1.e > f2.e) {
        e = f1.e;
        de = e - f2.e;
        m1 = (f1.m | M_MAX);
        m2 = (f2.m | M_MAX) >> de;
    }
    // e1 == e2
    else {
        e = f1.e;
        m1 = (f1.m | M_MAX);
        m2 = (f2.m | M_MAX);
    }

    if (m1 > m2) {
        s = f1.s;
        m = m1 - m2;
    }
    else {
        s = f1.s ^ 1;
        m = m2 - m1;
    }

    while(m <= M_BTM) {
        e -= 1;
        m <<= 1;
    }
    m ^= M_MAX;

    r.s = s;
    r.e = e;
    r.m = m;
    return r;

}

/*
 * al, ah:   8 bit
 * ax       16 bit
 * eax      32 bit
 *
 * mov  eax, v1
 * mul  v2          -> edx:eax
 * mov  r, edx
 */

uint64_t mul64(uint32_t m1, uint32_t m2) {
    union {
        struct {uint32_t lo, hi; } u;
        uint64_t m;
    };

    // asm (
    //     "mov eax, %1;"
    //     "mul %2;"
    //     "mov [%0+0], eax;"
    //     "mov [%0+4], edx;"
    //     : "=m"(u.lo)
    //     : "m"(m1),"m"(m2)
    //     : /**/
    // );
    asm (
        "mov eax, %1;"
        "mul %2;"       // -> edx:eax
        "shl rdx, 32;"  // -> rdx[hi] = edx
        "or  rdx, rax;" // -> rdx[lo] = eax
        "mov %0, rdx"   // -> rdx -> 'm'
        : "=m"(m)
        : "m"(m1),"m"(m2)
        : /**/
        );
    return m;
}

uint32_t div32old(uint32_t m1, uint32_t m2) {
    uint32_t m;

    asm (
        "mov eax, %1;"
        "mov edx, 0;"
        "div %2;"
        "mov %0, eax;"
        : "=m"(m)
        : "m"(m1),"m"(m2)
        : /**/
        );
    return m;
}

uint32_t div32(uint64_t m1, uint32_t m2) {
    union {
        struct {uint32_t lo, hi; };
        uint64_t m;
    } t1;
    uint32_t m;
    t1.m = m1;

    asm (
        "mov eax, %1;"
        "mov edx, %2;"
        "div %3;"
        "mov %0, eax;"
        : "=m"(m)
        : "m"(t1.lo), "m"(t1.hi), "m"(m2)
        : /**/
        );
    return m;
}

uint64_t mul64sh23(uint32_t m1, uint32_t m2) {
    union {
        struct {uint32_t lo, hi; } u;
        uint64_t m;
    };

    // asm (
    //     "mov eax, %1;"
    //     "mul %2;"
    //     "mov [%0+0], eax;"
    //     "mov [%0+4], edx;"
    //     : "=m"(u.lo)
    //     : "m"(m1),"m"(m2)
    //     : /**/
    // );
    asm (
        "mov eax, %1;"
        "mul %2;"       // -> edx:eax
        "shl rdx, 32;"  // -> rdx[hi] = edx
        "or  rdx, rax;" // -> rdx[lo] = eax
        "shr rdx, 23;" // -> rdx >>= 23
        "mov %0, rdx"   // -> rdx -> 'm'
        : "=m"(m)
        : "m"(m1),"m"(m2)
        : /**/
        );
    return m;
}

uint32_t div32sh23(uint32_t m1, uint32_t m2) {
    uint32_t m;

    // asm (
    //     "mov eax, %1;"
    //     "mov edx, 0;"
    //     "div %2;"
    //     "shl rax, 23;"
    //     "mov %0, eax;"
    //     : "=m"(m)
    //     : "m"(m1),"m"(m2)
    //     : /**/
    //     );

    asm (
        "mov eax, %1;"
        "mov edx, 0;"
        "div %2;"       // eax=q, edx=r
        "shl eax, 23;"
        "mov %0, eax;"
        : "=m"(m)
        : "m"(m1),"m"(m2)
        : /**/
        );
    return m;
}



float32 ieee754::mul(const float32& f1, const float32& f2)  {
    uint32_t s, e, m1, m2, m;
    float32 r;

    s  = f1.s != f2.s;
    e  = f1.e + f2.e - E_OFS;
    m1 = f1.m | M_MAX;
    m2 = f2.m | M_MAX;
    // m = mul64(m1, m2) >> M_LEN;
    m = mul64sh23(m1, m2);

    while(m >= M_TOP) {
        e += 1;
        m >>= 1;
    }
    while(m > 0 && m < M_BTM) {
        e -= 1;
        m <<= 1;
    }

    m ^= M_MAX;

    r.s = s;
    r.e = e;
    r.m = m;

    return r;
}

// float32 float32::div(const float32& f1, const float32& f2) {
//     uint32_t s, e, m2, m;
//     float32 r;
//     uint64_t m1;
//
//     s  = f1.s ^ f2.s;
//     e  = f1.e - f2.e + E_OFS;
//
//     m1 = f1.m | M_MAX;
//     m2 = f2.m | M_MAX;
//     m1 <<= M_LEN;
//     m = m1/m2;
//
//     while(m >= M_TOP) {
//         e += 1;
//         m >>= 1;
//     }
//     while(m > 0 && m < M_BTM) {
//         e -= 1;
//         m <<= 1;
//     }
//
//     m ^= M_MAX;
//
//     r.s = s;
//     r.e = e;
//     r.m = m;
//
//     return r;
// }


float32 ieee754::div(const float32& f1, const float32& f2)  {
    uint32_t s, e, m2, m;
    float32 r;
    uint64_t m1;

    s  = f1.s != f2.s;
    e  = E_OFS + f1.e - f2.e;
    m1 = f1.m | M_MAX;
    m2 = f2.m | M_MAX;
    m1 <<= M_LEN;
    m = div32(m1, m2);

    while(m >= M_TOP) {
        e += 1;
        m >>= 1;
    }
    while(m > 0 && m < M_BTM) {
        e -= 1;
        m <<= 1;
    }

    m ^= M_MAX;

    r.s = s;
    r.e = e;
    r.m = m;

    return r;
}


int ieee754::cmp(const float32& f1, const float32& f2)
{
    bool s = f1.s;
    //          -      +      |  -/+     -/+     | + -
    int cs = f1.s > f2.s ? -1 : f1.s == f2.s ? 0 : +1;
    if (cs != 0) return cs;

    int ce = f1.e < f2.e ? -1 : f1.e == f2.e ? 0 : +1;
    if (ce != 0)
        return s ? -ce : ce;

    int cm = f1.m < f2.m ? -1 : f1.m == f2.m ? 0 : +1;
    if (cm != 0)
        return s ? -cm : cm;

    return 0;
}
