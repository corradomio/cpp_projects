#include <iostream>
#include <intrin.h>
#include <x86gprintrin.h>
//using namespace std;
#include <math.h>

// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#
// https://gcc.gnu.org/onlinedocs/gcc/x86-Built-in-Functions.html

int main() {
    const char* fea = "fma";

    __builtin_cpu_init();
    printf("cmov    %d\n", __builtin_cpu_supports("cmov"));
    printf("mmx     %d\n", __builtin_cpu_supports("mmx"));
    printf("popcnt  %d\n", __builtin_cpu_supports("popcnt"));
    printf("sse     %d\n", __builtin_cpu_supports("sse"));
    printf("sse2    %d\n", __builtin_cpu_supports("sse2"));
    printf("sse3    %d\n", __builtin_cpu_supports("sse3"));
    printf("ssse3   %d\n", __builtin_cpu_supports("ssse3"));
    printf("sse4.1  %d\n", __builtin_cpu_supports("sse4.1"));
    printf("sse4.2  %d\n", __builtin_cpu_supports("sse4.2"));
    printf("sse4a   %d\n", __builtin_cpu_supports("sse4a"));
    printf("avx     %d\n", __builtin_cpu_supports("avx"));
    printf("avx2    %d\n", __builtin_cpu_supports("avx2"));

/*
    AVX
        AVX2
        AVX-128
        AVX-256
        AVX-512
    AVX512
        f, cd, er, pf  vl, dq, bw
        ifma, vbmi 4vnniw, 4fmaps
        vpopcntdq vnni, vbmi2, bitalg vp2intersect
        gfni, vpclmulqdq, vaes
        bf16, fp16
 */
    printf("avx512f  %d\n", __builtin_cpu_supports("avx512f"));
    printf("avx512cd %d\n", __builtin_cpu_supports("avx512cd"));
    printf("avx512er %d\n", __builtin_cpu_supports("avx512er"));
    printf("avx512pf %d\n", __builtin_cpu_supports("avx512pf"));
    printf("avx512vl %d\n", __builtin_cpu_supports("avx512vl"));
    printf("avx512dq %d\n", __builtin_cpu_supports("avx512dq"));
    printf("avx512bw %d\n", __builtin_cpu_supports("avx512bw"));
    printf("avx512ifma %d\n", __builtin_cpu_supports("avx512ifma"));
    printf("avx512vbmi %d\n", __builtin_cpu_supports("avx512vbmi"));
    printf("avx5124vnniw %d\n", __builtin_cpu_supports("avx5124vnniw"));
    printf("avx5124fmaps %d\n", __builtin_cpu_supports("avx5124fmaps"));
    printf("avx512vpopcntdq %d\n", __builtin_cpu_supports("avx512vpopcntdq"));
    printf("avx512vnni %d\n", __builtin_cpu_supports("avx512vnni"));
    printf("avx512vbmi2 %d\n", __builtin_cpu_supports("avx512vbmi2"));
    printf("avx512bitalg %d\n", __builtin_cpu_supports("avx512bitalg"));
    printf("avx512vp2intersect %d\n", __builtin_cpu_supports("avx512vp2intersect"));
//    printf("avx512gfni %d\n", __builtin_cpu_supports("avx512gfni"));
//    printf("avx512vpclmulqdq %d\n", __builtin_cpu_supports("avx512vpclmulqdq"));
//    printf("avx512vaes %d\n", __builtin_cpu_supports("avx512vaes"));
    printf("avx512bf16 %d\n", __builtin_cpu_supports("avx512bf16"));
//    printf("avx512fp16 %d\n", __builtin_cpu_supports("avx512fp16"));

    printf("fma     %d\n", __builtin_cpu_supports("fma"));
    printf("fma4    %d\n", __builtin_cpu_supports("fma4"));

    printf("aes     %d\n", __builtin_cpu_supports("aes"));
    printf("xop     %d\n", __builtin_cpu_supports("xop"));

    printf("bmi     %d\n", __builtin_cpu_supports("bmi"));
    printf("bmi2    %d\n", __builtin_cpu_supports("bmi2"));
    printf("pclmul  %d\n", __builtin_cpu_supports("pclmul"));
    printf("gfni    %d\n", __builtin_cpu_supports("gfni"));
    printf("vpclmulqdq  %d\n", __builtin_cpu_supports("vpclmulqdq"));

    printf("x86-64  %d\n", __builtin_cpu_supports("x86-64"));
    printf("x86-64-v2  %d\n", __builtin_cpu_supports("x86-64-v2"));
    printf("x86-64-v3  %d\n", __builtin_cpu_supports("x86-64-v3"));
    printf("x86-64-v4  %d\n", __builtin_cpu_supports("x86-64-v4"));

    return (0);
}


int main1() {
    std::cout << "Hello, World!" << std::endl;
    unsigned short us[3] = {0, 0xFF, 0xFFFF};
    unsigned short usr;
    unsigned int   ui[4] = {0, 0xFF, 0xFFFF, 0xFFFFFFFF};
    unsigned int   uir;

    for (int i=0; i<3; i++) {
        usr = __lzcnt16(us[i]);
        std::cout << "__lzcnt16(0x" << std::hex << us[i] << ") = " << std::dec << usr << std::endl;
    }

    for (int i=0; i<4; i++) {
        uir = __lzcnt64(ui[i]);
        std::cout << "__lzcnt(0x" << std::hex << ui[i] << ") = " << std::dec << uir << std::endl;
    }

    return 0;
}
