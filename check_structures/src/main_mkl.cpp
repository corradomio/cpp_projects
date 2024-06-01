//
// Created by Corrado Mio on 21/03/2024.
//
#include <chrono>
#include <cstdio>
#include <mkl.h>
// #include <oneapi/mkl.hpp>
#include "stdx/float64/linalg.h"

using namespace stdx::float64;


int main_mkl1() {
    printf("Hello cruel world\n");
    MKLVersion mkl_version;
    mkl_get_version(&mkl_version);
    printf("You are using oneMKL %d.%d\n", mkl_version.MajorVersion, mkl_version.UpdateVersion);

    vector_t u = range(1000);
    vector_t v = range(1000);

    // 333833500
    // printf("%f\n", dot(u,v));
    real_t s = cblas_ddot(
            int(u.size()),
            u.data(),
            1,
            v.data(),
            1
    );
    printf("%f\n", s);

    matrix_t a = range(1000, 500);
    matrix_t b = range(500, 1000);
    matrix_t r(1000, 1000);

    bool tra = false;
    bool trb = false;

    printf("start\n");
    cblas_dgemm(
            CBLAS_LAYOUT::CblasRowMajor,
            tra ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
            trb ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
            int(tra ? a.cols() : a.rows()),
            int(trb ? b.rows() : b.cols()),
            int(tra ? a.rows() : a.cols()),
            1.0,
            a.data(),
            int(a.cols()),
            b.data(),
            int(b.cols()),
            0.0,
            r.data(),
            int(r.cols())
    );
    printf("end\n");
    printf("%f\n", frobenius(r));


    return 0;
}
