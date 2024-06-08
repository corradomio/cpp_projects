//
// Created by Corrado Mio on 19/04/2024.
//

#ifndef STDX_FLOAT64_SVD_H
#define STDX_FLOAT64_SVD_H

#include <tuple>
#include <stdx/options.h>
#include "vector.h"
#include "matrix.h"


namespace stdx::float64 {

    /// Retrieve the largest eigenvalue and eigenvector using the Power method
    ///
    /// Options:
    ///        "eps": 1e-8
    ///      "niter": 1000
    ///     "method": "power"
    std::pair<real_t, vector_t> largest_eigenval(const matrix_t& m, const stdx::options_t& opts);

    /// Retrieve the largest eigenvalue and eigenvector using the Power method
    ///
    /// Options:
    ///        "eps": 1e-8
    ///      "niter": 1000
    ///     "method": "simple", "schur", "divide_and_conquer"
    std::tuple<matrix_t, vector_t, matrix_t> svd(const matrix_t& m, const stdx::options_t& opts);

}

#endif //STDX_FLOAT64_SVD_H
