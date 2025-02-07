//
// Created by Corrado Mio on 26/03/2023.
//

#include <string>
#include <stdio.h>

//
// const_cast<type> (expr)
// dynamic_cast<type> (expr)
// reinterpret_cast<type> (expr)
// static_cast<type> (expr)
//

/*
 * Che strano:
 */

int main9()
{
    const char* fea = "fma";

    __builtin_cpu_init();
    printf("mmx     %d\n", __builtin_cpu_supports("mmx"));
    printf("sse     %d\n", __builtin_cpu_supports("sse"));
    printf("sse2    %d\n", __builtin_cpu_supports("sse2"));
    printf("sse3    %d\n", __builtin_cpu_supports("sse3"));
    printf("ssse3   %d\n", __builtin_cpu_supports("ssse3"));
    printf("sse41   %d\n", __builtin_cpu_supports("sse4.1"));
    printf("sse42   %d\n", __builtin_cpu_supports("sse4.2"));
    printf("avx     %d\n", __builtin_cpu_supports("avx"));
    printf("avx2    %d\n", __builtin_cpu_supports("avx2"));
//    printf("avx512  %d\n", __builtin_cpu_supports("avx512"));
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

    printf("aes     %d\n", __builtin_cpu_supports("aes"));
    printf("sha     %d\n", __builtin_cpu_supports("sha"));

//    printf("%d\n", __builtin_cpu_supports ("sse"));
//    printf("%d\n", __builtin_cpu_supports ("avx"));
//    printf("%d\n", __builtin_cpu_supports ("mmx"));

    return (0);
}
