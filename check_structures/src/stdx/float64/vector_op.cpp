//
// Created by Corrado Mio on 08/03/2024.
//
#include <iostream>
#include "stdx//float64/array_op.h"
#include "stdx//float64/vector_op.h"

namespace stdx::float64 {

    // ----------------------------------------------------------------------

    vector_t zeros(size_t n) {
        vector_t v{n};
        v = 0;
        return v;
    }

    vector_t zeros_like(const vector_t& v) {
        return zeros(v.size());
    }


    vector_t  ones(size_t n) {
        vector_t v{n};
        v = 1;
        return v;
    }

    vector_t range(size_t n) {
        vector_t v{n};
        for(size_t i=0; i<n; ++i)
            v[i] = 1.+i;
        return v;
    }

    vector_t uniform(size_t n, real_t min, real_t max) {
        real_t delta = max-min;
        vector_t v{n};
        for(size_t i=0; i < n; ++i)
            v[i] = min+(::rand()*delta)/RAND_MAX;
        return v;
    }

    // ----------------------------------------------------------------------

    bool operator == (const vector_t& u, const vector_t& v) {
        if (u.size() != v.size())
            return false;

        size_t n=u.size();
        for (size_t i=0; i<n; ++i)
            if (u[i] != v[i])
                return false;
        return true;
    }

    real_t min(const vector_t& v) {
        return min(static_cast<const array_t&>(v));
    }
    real_t max(const vector_t& v) {
        return max(static_cast<const array_t&>(v));
    }

    // ----------------------------------------------------------------------

    void print(const vector_t& v) {
        size_t n = v.size();
        std::cout << "[ ";
        for(int i=0; i<n; ++i)
            std::cout << v[i] << " ";
        std::cout << "]" << std::endl;
    }
}