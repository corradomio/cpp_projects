//
// Created by Corrado Mio on 17/03/2024.
//
#include <tuple>
#include "stdx/options.h"
#include "stdx/float64/matrix.h"
#include "stdx/float64/matrix_op.h"
#include "stdx/float64/dot_op.h"
#include "stdx/float64/matrix_factorization.h"

using namespace stdx;
using namespace stdx::float64;


int main() {

    matrix_t V = uniform(10, 5, 0., 10.);
    print(V);

    options_t opts = options_t()
        .set("eps", 1.e-12)
        .set("niter", 500000)
        .set("verbose", false);

    std::tuple<matrix_t, matrix_t> WH = nmf(V, 5, opts);

    matrix_t W = std::get<0>(WH);
    matrix_t H = std::get<1>(WH);

    // print(V);
    print(dot(W,H));
    printf("%g\n", frobenius(V, dot(W,H)));

    // print(chop(V-dot(W,H)));

    return 0;
}