//
// Created by Corrado Mio on 21/03/2024.
//
#include <chrono>
#include <cstdio>
#include <iostream>
#include <openblas/cblas.h>
#include "stdx/tprint.h"
#include "stdx/float64/linalg.h"

using namespace stdx::float64;


int main74() {
    matrix_t m1 = range(5,4);
    matrix_t m2 = range(4,5);

    matrix_t r = dot(m1, m2);
    print(r);

    return 0;
}


int main73() {

    matrix_t m = range(5,4);
    // vector_t u = range(4);
    // vector_t r = dot(m, u);
    vector_t u = range(5);
    vector_t r = dot(u, m);

    print(m);
    print(u);
    print(r);

    return 0;
}


int main72() {
    typedef std::chrono::high_resolution_clock Clock;

    stdx::tprint(); printf("allocate\n");
    matrix_t a = range(1000, 500);
    matrix_t b = range(500, 1000);
    matrix_t r = zeros(1000, 1000);

    bool tra = false;
    bool trb = false;

    // stdx::tprint();
    printf("start\n");
    auto t1 = Clock::now();
    // cblas_dgemm(
    //         CBLAS_LAYOUT::CblasRowMajor,
    //         tra ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
    //         trb ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
    //         int(tra ? a.cols() : a.rows()),
    //         int(trb ? b.rows() : b.cols()),
    //         int(tra ? a.rows() : a.cols()),
    //         1.0,
    //         a.values(),
    //         int(a.cols()),
    //         b.values(),
    //         int(b.cols()),
    //         0.0,
    //         r.values(),
    //         int(r.cols())
    // );
    dot_eq(r, a, b);
    printf("end\n");
    auto t2 = Clock::now();
    std::cout << std::chrono::duration<double, std::milli>(t2-t1) <<  std::endl;

    printf("%f\n", frobenius(r));
    return 0;
}


int main71() {

    stdx::tprint(); printf("allocate\n");
    // vector_t u  = range(1000);
    // matrix_t m1 = range(1000, 500);
    // matrix_t m2 = range(500, 1000);

    vector_t u  = range(10);
    matrix_t m1 = range(5, 10);
    matrix_t m2 = range(10, 5);

    stdx::tprint(); printf("start\n");
    // printf("%g\n", dot(u,u));
    // print(dot(m1,u));
    // print(dot(u, m2));
    stdx::tprint(); printf("end\n");

    return 0;
}