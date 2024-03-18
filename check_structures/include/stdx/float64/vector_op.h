//
// Created by Corrado Mio on 08/03/2024.
//

#ifndef STDX_FLOAT64_VECTOR_OP_H
#define STDX_FLOAT64_VECTOR_OP_H

#include <stdexcept>
#include <algorithm>
#include "vector.h"

namespace stdx::float64 {

    vector_t zeros(size_t n);
    vector_t  ones(size_t n);
    vector_t range(size_t n);
    vector_t uniform(size_t n, real_t min=0., real_t max=1.);
    vector_t zeros_like(const vector_t& v);

    bool operator == (const vector_t& u, const vector_t& v);

    real_t min(const vector_t& v);
    real_t max(const vector_t& v);

    void print(const vector_t& v);
}

#endif //STDX_FLOAT64_VECTOR_OP_H
