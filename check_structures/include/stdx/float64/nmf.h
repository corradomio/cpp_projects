//
// Created by Corrado Mio on 17/03/2024.
//

#ifndef STDX_FLOAT64_MATRIX_FACTORIZATION_H
#define STDX_FLOAT64_MATRIX_FACTORIZATION_H

#include <tuple>
#include "../options.h"
#include "matrix.h"

namespace stdx::float64 {

    /// Non Negative Matrix Factorization
    ///
    /// \param M matrix
    /// \param k low rank
    /// \param opts extra options
    ///     'eps' (1.e-8)   error in two consecutive iterations
    ///     'niter' (10000)  maximum number of iterations
    ///     'verbose' (false) if to show the errors
    /// \return the factor matrices
    std::tuple<matrix_t, matrix_t> nmf(const matrix_t& M, size_t k, const stdx::options_t& opts);

}

#endif //STDX_FLOAT64_MATRIX_FACTORIZATION_H
