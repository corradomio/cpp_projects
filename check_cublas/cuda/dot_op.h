//
// Created by Corrado Mio on 23/03/2024.
//

#ifndef CUDA_DOT_OP_H
#define CUDA_DOT_OP_H

#include "cublas.h"

namespace cuda {

    real_t dot(const vector_t& u, const vector_t& v);

    void dot_eq(vector_t& r, const matrix_t& m, const vector_t& v);
    void dot_eq(vector_t& r, const vector_t& u, const matrix_t& m);

    vector_t dot(matrix_t& m, const vector_t& v);
    vector_t dot(const vector_t& u, const matrix_t& m);

    void cross_eq(matrix_t& m, const vector_t& u, const vector_t& v);
    matrix_t cross(const vector_t& u, const vector_t& v);   // u x v == u.v^T

    // R = A.B | A^T.B | A.B^T | A^T.B^T  not supported
    void dot_eq(matrix_t& r, const matrix_t& a, const matrix_t& b, bool tra=false, bool trb=false);
    matrix_t dot(const matrix_t& a, const matrix_t& b, bool tra=false, bool trb=false);

}

#endif //CUDA_DOT_OP_H
