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

    // u[i] = f(u[i])
    void apply_eq(array_t& u, real_t (*f)(real_t));
    // u[i] = f(u[i], s)
    void apply_eq(array_t& u, real_t (*f)(real_t, real_t), real_t s);
    // u[i] = f(u[i], v[i])
    void apply_eq(array_t& u, real_t (*f)(real_t, real_t), const array_t& v);
    // u[i] = f(u[i], s, v[i])
    void apply_eq(array_t& u, real_t (*f)(real_t, real_t, real_t), real_t s, const array_t& v);
    // u[i] = f(v[i], w[i]);
    void apply_eq(array_t& u, real_t (*f)(real_t, real_t), const array_t& v, const array_t& w);

    // sum(i, f(u[i]))
    real_t reduce(real_t (*f)(real_t), const array_t& u);
    // sum(i, f(u[i],s)
    real_t reduce(real_t (*f)(real_t, real_t), const array_t& u, real_t s);
    // sum(i, f(u[i],v[i]))
    real_t reduce(real_t (*f)(real_t, real_t), const array_t& u, const array_t& v);

    // sum(i, f(u[ou+i*su],v[ov+i*sv])
    real_t reduce(real_t (*f)(real_t, real_t), const array_t& u, const array_t& v,
                  size_t n,
                  size_t ou, size_t su,     // offset/skip u
                  size_t ov, size_t sv);    // offset/skip v

    real_t min(const array_t& v);
    real_t max(const array_t& v);
}

#endif //STDX_FLOA64_ARRAY_OP_H
