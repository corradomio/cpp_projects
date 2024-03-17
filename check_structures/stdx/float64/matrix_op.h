//
// Created by Corrado Mio on 08/03/2024.
//

#ifndef STDX_FLOAT64_MATRIX_OP_H
#define STDX_FLOAT64_MATRIX_OP_H

#include "matrix.h"

namespace stdx::float64 {

    matrix_t zeros(size_t nr, size_t nc);
    matrix_t  ones(size_t nr, size_t nc);
    matrix_t range(size_t nr, size_t nc);
    matrix_t identity(size_t nr, size_t nc=-1);
    matrix_t uniform(size_t nr, size_t nc, real_t min=0, real_t max=1);

    bool operator == (const matrix_t& a, const matrix_t& b);

    real_t norm(const matrix_t& m, int p);
    real_t frobenius(const matrix_t& m);

    void print(const matrix_t& m);
}

#endif //STDX_FLOAT64_MATRIX_OP_H
