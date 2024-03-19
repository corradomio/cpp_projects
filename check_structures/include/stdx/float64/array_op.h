//
// Created by Corrado Mio on 08/03/2024.
//

#include <stdexcept>
#include "array.h"

#ifndef STDX_FLOA64_ARRAY_OP_H
#define STDX_FLOA64_ARRAY_OP_H

#include "arith.h"

namespace stdx::float64 {

    void check(const array_t& u, const array_t& v);

    // u = f(u)
    void apply_eq(array_t& u, real_t (*f)(real_t));
    // u = f(u, s)
    void apply_eq(array_t& u, real_t (*f)(real_t, real_t), real_t s);
    // u = f(u, v)
    void apply_eq(array_t& u, real_t (*f)(real_t, real_t), const array_t& v);
    // u = f(u, s, v)
    void apply_eq(array_t& u, real_t (*f)(real_t, real_t, real_t), real_t s, const array_t& v);
    // u = f(v, w);
    void apply_eq(array_t& u, real_t (*f)(real_t, real_t), const array_t& v, const array_t& w);

    real_t reduce(array_t& u, real_t (*f)(real_t));
    real_t reduce(array_t& u, real_t (*f)(real_t, real_t), real_t s);
    real_t reduce(array_t& u, real_t (*f)(real_t, real_t), const array_t& v);

    real_t min(const array_t& v);
    real_t max(const array_t& v);
}

#endif //STDX_FLOA64_ARRAY_OP_H
