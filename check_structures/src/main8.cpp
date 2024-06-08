//
// Created by Corrado Mio on 18/04/2024.
//
#include <chrono>
#include <cstdio>
#include <iostream>
#include "stdx/float64/linalg.h"

using namespace stdx::float64;

int main8() {
    matrix_t m = uniform(5,4, 1., 10.);
    print(m, array_style::MATHEMATICA);

    static stdx::options_t opts = stdx::options_t()
        ("eps", 1.e-8)
        ("niter", 0)
        ("tr", false)
    ;

    std::tuple<matrix_t, vector_t, matrix_t> udvt = svd(m, opts);
    matrix_t u = std::get<0>(udvt);
    vector_t d = std::get<1>(udvt);
    matrix_t vt = std::get<2>(udvt);

    printf("--\n");

    print(m, array_style::MATHEMATICA);

    printf("--\n");

    print(u, array_style::MATHEMATICA);
    print(d, array_style::MATHEMATICA);
    print(vt, array_style::MATHEMATICA);
}

int main81() {
    printf("Hello cruel world\n");

    matrix_t m1 = uniform(30,30, 1., 10.);

    print(m1, array_style::PYTHON);

    static stdx::options_t opts_1 = stdx::options_t()
        ("eps", 1.e-8)
        ("niter", 0)
        ("positive", true)
        ("method", "power")
    ;

    std::pair<real_t, vector_t> eigen;
    real_t   eval;
    vector_t evec;

    eigen = largest_eigenval(m1, opts_1);
    eval = get<0>(eigen);
    evec = get<1>(eigen);

    printf("--\n");
    printf("%f\n", eval);
    print(evec, array_style::PYTHON);

    printf("--\n");
    return 0;
}


// int main81() {
//     matrix_t m1 = uniform(10,5, 1., 10.);
//
//     static stdx::options_t opts = stdx::options_t()
//             ("eps", 1.e-8)
//             ("niter", 0);
//     std::tuple<matrix_t, matrix_t> uv = nmf(m1, 3, opts);
//     matrix_t u = get<0>(uv);
//     matrix_t v = get<1>(uv);
//
//     print(m1);
//     print(u);
//     print(v);
//     print(m1 - dot(u,v));
//
//     return 0;
// }

