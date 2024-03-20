//
// Created by Corrado Mio on 17/03/2024.
//
#include <tuple>
#include "stdx/options.h"
#include "stdx/float64/vector.h"
#include "stdx/float64/vector_op.h"
#include "stdx/float64/matrix.h"
#include "stdx/float64/matrix_op.h"
#include "stdx/float64/dot_op.h"
#include "stdx/float64/nmf.h"

using namespace stdx;
using namespace stdx::float64;

int main52() {
    // matrix_t M = range(5, 3);
    // print(M);
    // print(tr(M));

    matrix_t M = range(3,3);
    vector_t v = range(3);

    print(M);
    print(v);

    matrix_t R1 = ddot(M, v);
    matrix_t R2 = ddot(v, M);
    // matrix_t R3 = ddot(M, v, true);
    // matrix_t R4 = ddot(v, M, true);
    //
    print(R1);
    print(R2);
    // print(R3);
    // print(R4);

    return 0;
}


int main() {

    // matrix_t V = uniform(10, 5, 0., 10.);
    // print(V);
    printf("Create matrix\n");
    matrix_t V = uniform(1000, 250, 10., 100.);

    options_t opts = options_t()
        .set("eps", 1.e-12)
        .set("niter", 500000)
        .set("verbose", 100);

    printf("NMF\n");
    std::tuple<matrix_t, matrix_t> WH = nmf(V, 100, opts);

    matrix_t W = std::get<0>(WH);
    matrix_t H = std::get<1>(WH);

    // print(V);
    // print(dot(W,H));
    // printf("%g\n", frobenius(V, dot(W,H)));

    // print(chop(V-dot(W,H)));

    return 0;
}