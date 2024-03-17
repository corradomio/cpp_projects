//
// Created by Corrado Mio on 08/03/2024.
//

#ifndef STDX_FLOAT64_DOT_OP_H
#define STDX_FLOAT64_DOT_OP_H

#include "vector.h"
#include "matrix.h"

namespace stdx::float64 {

    real_t   dot(const vector_t& u, const vector_t& v);       // u.v
    vector_t dot(const matrix_t& m, const vector_t& v);       // M.v
    vector_t dot(const vector_t& u, const matrix_t& m);       // u.M
    matrix_t dot(const matrix_t& a, const matrix_t& b);       // A.B

    // R = A.B
    // R = A^T.B
    // R = A.B^T
    // R = A^T.B^T  not supported
    void dot_eq(matrix_t& r, const matrix_t& a, const matrix_t& b, bool tra=false, bool trb=false);

    // A^T.B
    matrix_t tdot(const matrix_t& a, const matrix_t& b);
    // A.B^T
    matrix_t dott(const matrix_t& a, const matrix_t& b);

    // u x v
    matrix_t cross(const vector_t& u, const vector_t& v);

}

#endif //STDX_FLOAT64_DOT_OP_H
