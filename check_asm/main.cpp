#include <iostream>
#include <cstdint>
#include <stdfloat>
#include <stdio.h>

// MUL
// DIV
// SAL/SAR  r/m, 1/i/cl     Shift Arithmetic Left/Right     1/immediate value/cl registry
// SHL/SHR  r/m, 1/i/cl     Shift Logical    Left/Right

// uint64_t mul64Old(uint32_t m1, uint32_t m2) {
//     union {
//         struct {uint32_t lo, hi; } u;
//         uint64_t m;
//     };
//
//     asm (
//         "mov eax, %2;"
//         "mul %3;"
//         "mov %0, eax;"
//         "mov %1, edx;"
//         // "mov %1, edx;"
//         : "=m"(u.lo), "=m"(u.hi)
//         : "m"(m1),"m"(m2)
//         : /**/
//     );
//     return m;
// }


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


uint32_t mul(uint32_t m1, uint32_t m2) {
    uint32_t m = 1;
    asm (
            "mov eax, %1;"
            "mul %2;"
            "mov %0, eax;"
            : "=m"(m)
            : "m"(m1),"m"(m2)
            : /**/
            );
    return m;
}


int main() {

    union float32_t {
        float f;
        struct { unsigned m:23; unsigned e:8; unsigned s:1; };

        float32_t() { }
        float32_t(float f): f(f) {}
        float32_t(unsigned e, unsigned m): float32_t(0, e, m) {};
        float32_t(unsigned s, unsigned e, unsigned m): s(s), e(e), m(m) {
            printf("(%d, %2X, %7X} %.10g\n", s, e, m, f);
        };

    };

    float32_t f10(0,1<<0);
    // float32_t f11(0,1<<1);
    // float32_t f122(0,1<<22);
    float32_t f2(0,0x7FFFFFF);
    float32_t f3(1,0);
    float32_t f4(126,0);
    float32_t f5(127,0);

    // std::cout << f1.f << std::endl;
    // std::cout << f2.f << std::endl;
    // std::cout << f3.f << std::endl;
    // std::cout << f4.f << std::endl;

}


int main1() {

    uint32_t I16  = (1L << 16) - 1;
    uint64_t L16  = (1LL << 16) - 1LL;
    uint64_t I31  = (1LL << 31) - 1;
    uint32_t MAX = 0xFFFFFFFFul;
    uint64_t MAX64 = 0xFFFFFFFFFFFFFFFFull;
    uint64_t RES = 18446744065119617025ull;

    // std::cout << MAX << std::endl;
    // std::cout << I16 << std::endl;
    std::cout << mul(I16,I16) << " " << (I16*I16) << std::endl;
    std::cout << mul64(I16,I16) << " " << ((uint64_t)(L16*L16)) << std::endl;
    std::cout << std::hex << mul64(I31,I31) << " " << (I31*I31) << std::endl;

    // std::cout << mul(65535,65535) << " = " << (65535u*65535u) << std::endl;
    // std::cout << mul(MAX,MAX) << " = " << (MAX64) << std::endl;
    // std::cout << mul(3,2) << std::endl;
    // std::cout << mul64(3,2) << std::endl;
    // std::cout << std::hex << mul64(MAX,MAX) << std::endl;
    // std::cout << (RES) << std::endl;
    // std::cout << "Hello, World!" << std::endl;
    return 0;
}
