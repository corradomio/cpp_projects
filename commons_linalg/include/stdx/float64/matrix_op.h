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
    matrix_t uniform(size_t nr, size_t nc, real_t min, real_t max);
    matrix_t zeros_like(const matrix_t& m);

    inline matrix_t like(const matrix_t& m) {
        return matrix_t{ m.rows(), m.cols() };
    }

    bool operator == (const matrix_t& a, const matrix_t& b);

    real_t min(const matrix_t& m);
    real_t max(const matrix_t& m);

    real_t norm(const matrix_t& m, int p=2);
    real_t frobenius(const matrix_t& m);
    real_t frobenius(const matrix_t& a, const matrix_t& b);

    // R = A + B | A - B | A*B | A/B
    void neg_eq(matrix_t& r);
    void sum_eq(matrix_t& r, const matrix_t& a, const matrix_t& b);
    void sub_eq(matrix_t& r, const matrix_t& a, const matrix_t& b);
    void mul_eq(matrix_t& r, const matrix_t& a, const matrix_t& b);
    void div_eq(matrix_t& r, const matrix_t& a, const matrix_t& b);

    matrix_t sum(const matrix_t& a, const matrix_t& b);
    matrix_t sub(const matrix_t& a, const matrix_t& b);
    matrix_t mul(const matrix_t& a, const matrix_t& b);
    matrix_t div(const matrix_t& a, const matrix_t& b);

    inline matrix_t operator + (const matrix_t& a) { matrix_t r(a);                return r; }
    inline matrix_t operator - (const matrix_t& a) { matrix_t r(a); neg_eq(r); return r; }
    inline matrix_t operator + (const matrix_t& a, const matrix_t& b) { return sum(a, b); }
    inline matrix_t operator - (const matrix_t& a, const matrix_t& b) { return sub(a, b); }
    inline matrix_t operator * (const matrix_t& a, const matrix_t& b) { return mul(a, b); }
    inline matrix_t operator / (const matrix_t& a, const matrix_t& b) { return div(a, b); }

    // r = a + s*b
    void linear_eq(matrix_t& r, const matrix_t& a, real_t s, const matrix_t& b);

    matrix_t chop(const matrix_t& m, real_t eps=1.e-8);

    // transpose
    matrix_t tr(const matrix_t& m);

    void print(const matrix_t& m, array_style style=array_style::PYTHON);
    void print_dim(const char* name, const matrix_t& m);
}

#endif //STDX_FLOAT64_MATRIX_OP_H
