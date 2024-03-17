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

    bool operator == (const vector_t& u, const vector_t& v);
    void print(const vector_t& v);
}

#endif //STDX_FLOAT64_VECTOR_OP_H
