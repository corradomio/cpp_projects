//
// Created by Corrado Mio on 08/03/2024.
//

#ifndef STDX_FLOAT64_VECTOR_OP_H
#define STDX_FLOAT64_VECTOR_OP_H

#include <stdexcept>
#include <algorithm>
#include "vector.h"

namespace stdx::float64 {

    vector_t   zeros(size_t n);
    vector_t    ones(size_t n);
    vector_t   range(size_t n);
    vector_t uniform(size_t n, real_t min, real_t max);
    inline vector_t like(const vector_t& v) { return vector_t{ v.size() }; }

    inline void swap(vector_t& u, vector_t& v) {
        vector_t t = u;
        u = v;
        v = t;
    }

    bool operator == (const vector_t& u, const vector_t& v);

    real_t min(const vector_t& v);
    real_t max(const vector_t& v);

    real_t norm(const vector_t& v, int p=2);
    real_t norm(const vector_t& u, const vector_t& v, int p=2);

    // R = A + B | A - B | A*B | A/B
    inline void sum_eq(vector_t& r, const vector_t& u, const vector_t& v) { apply_eq(r, sum, u, v); }
    inline void sub_eq(vector_t& r, const vector_t& u, const vector_t& v) { apply_eq(r, sub, u, v); }
    inline void mul_eq(vector_t& r, const vector_t& u, const vector_t& v) { apply_eq(r, mul, u, v); }
    inline void div_eq(vector_t& r, const vector_t& u, const vector_t& v) { apply_eq(r, div, u, v); }

    inline vector_t sum(const vector_t& u, const vector_t& v) { vector_t r(u.size()); sum_eq(r, u, v); return r; }
    inline vector_t sub(const vector_t& u, const vector_t& v) { vector_t r(u.size()); sub_eq(r, u, v); return r; }
    inline vector_t mul(const vector_t& u, const vector_t& v) { vector_t r(u.size()); mul_eq(r, u, v); return r; }
    inline vector_t div(const vector_t& u, const vector_t& v) { vector_t r(u.size()); div_eq(r, u, v); return r; }

    inline void neg_eq(vector_t& r) { apply_eq(r, neg); }
    // inline void abs_eq(vector_t& r) { apply_eq(r, abs); }
    inline void mul_eq(vector_t& r, real_t s) { apply_eq(r, mul, s); }
    inline void div_eq(vector_t& r, real_t s) { apply_eq(r, div, s); }

    inline vector_t neg(const vector_t& u) { vector_t r(u, true); neg_eq(r); return r; }
    // inline vector_t abs(const vector_t& u, real_t s) { vector_t r(u, true); abs_eq(r); return r; }
    inline vector_t mul(const vector_t& u, real_t s) { vector_t r(u, true); apply_eq(r, mul, s); return r; }
    inline vector_t div(const vector_t& u, real_t s) { vector_t r(u, true); apply_eq(r, div, s); return r; }

    inline vector_t operator + (const vector_t& u) { vector_t r(u);                return r; }
    inline vector_t operator - (const vector_t& u) { vector_t r(u); neg_eq(r); return r; }
    inline vector_t operator + (const vector_t& u, const vector_t& v) { return sum(u, v); }
    inline vector_t operator - (const vector_t& u, const vector_t& v) { return sub(u, v); }
    inline vector_t operator * (const vector_t& u, const vector_t& v) { return mul(u, v); }
    inline vector_t operator / (const vector_t& u, const vector_t& v) { return div(u, v); }

    // r = u + s*v
    void linear_eq(vector_t& r, const vector_t& u, real_t s, const vector_t& v);
    vector_t uversor(size_t n, real_t min, real_t max);

    void print(const vector_t& v, array_style style=array_style::PYTHON);
}

#endif //STDX_FLOAT64_VECTOR_OP_H
