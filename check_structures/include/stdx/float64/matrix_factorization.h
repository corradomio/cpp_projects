//
// Created by Corrado Mio on 17/03/2024.
//

#ifndef STDX_FLOAT64_MATRIX_FACTORIZATION_H
#define STDX_FLOAT64_MATRIX_FACTORIZATION_H

#include <tuple>
#include "../options.h"
#include "matrix.h"

namespace stdx::float64 {

    std::tuple<matrix_t, matrix_t> nmf(matrix_t M, size_t k, const stdx::options_t& opts);

}

#endif //STDX_FLOAT64_MATRIX_FACTORIZATION_H
