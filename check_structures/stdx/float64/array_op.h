//
// Created by Corrado Mio on 08/03/2024.
//

#include <stdexcept>
#include "array.h"

#ifndef STDX_FLOA64_ARRAY_OP_H
#define STDX_FLOA64_ARRAY_OP_H

namespace stdx::float64 {

    void check(const array_t& u, const array_t& v);

    void apply_eq(real_t (*f)(real_t), array_t& u);
    void apply_eq(real_t (*f)(real_t, real_t), array_t& u, real_t s);
    void apply_eq(real_t (*f)(real_t, real_t), array_t& u, const array_t& v);
    void apply_eq(real_t (*f)(real_t, real_t, real_t), array_t& u, real_t s, const array_t& v);

    real_t reduce(real_t (*f)(real_t), array_t& u);
    real_t reduce(real_t (*f)(real_t, real_t), array_t& u, real_t s);
    real_t reduce(real_t (*f)(real_t, real_t), array_t& u, const array_t& v);

    real_t min(const array_t& v);
    real_t max(const array_t& v);
}

#endif //STDX_FLOA64_ARRAY_OP_H
