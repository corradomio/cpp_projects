//
// Created by Corrado Mio on 17/03/2024.
//
#include <tuple>
#include "stdx/ref/map.h"
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
        .set("eps", 1.e-8)
        .set("niter", 100000);

    printf("%g\n", opts.get("eps", 0.));
    printf("%d\n", opts.get("niter", 0));
    fflush(stdout);

    std::tuple<matrix_t, matrix_t> WH = nmf(V, 5, opts);

    matrix_t W = std::get<0>(WH);
    matrix_t H = std::get<1>(WH);

    // print(V);
    print(dot(W,H));
    printf("%g\n", frobenius(V, dot(W,H)));

    return 0;
}