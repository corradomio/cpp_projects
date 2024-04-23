//
// Created by Corrado Mio on 18/04/2024.
//
#include <chrono>
#include <cstdio>
#include <iostream>
#include <openblas/cblas.h>
#include "stdx/tprint.h"
#include "stdx/float64/linalg.h"

using namespace stdx::float64;

int main() {
    matrix_t m1 = uniform(30,30, 1., 10.);

    print(m1);

    static stdx::options_t opts = stdx::options_t()
            ("eps", 1.e-8)
            ("niter", 0);


    std::pair<real_t, vector_t> eigen = largest_eigenval(m1, opts);
    real_t   eval = get<0>(eigen);
    vector_t evec = get<1>(eigen);

    printf("%f\n", eval);
    print(evec);

    return 0;
}


int main81() {
    matrix_t m1 = uniform(10,5, 1., 10.);

    static stdx::options_t opts = stdx::options_t()
            ("eps", 1.e-8)
            ("niter", 0);
    std::tuple<matrix_t, matrix_t> uv = nmf(m1, 3, opts);
    matrix_t u = get<0>(uv);
    matrix_t v = get<1>(uv);

    print(m1);
    print(u);
    print(v);
    print(m1 - dot(u,v));

    return 0;
}

