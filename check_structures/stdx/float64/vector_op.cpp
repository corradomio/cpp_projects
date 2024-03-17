//
// Created by Corrado Mio on 08/03/2024.
//
#include <iostream>
#include "vector_op.h"

namespace stdx::float64 {

    // ----------------------------------------------------------------------

    vector_t zeros(size_t n) {
        vector_t v{n};
        v = 0;
        return v;
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

    // ----------------------------------------------------------------------

    void print(const vector_t& v) {
        size_t n = v.size();
        std::cout << "[ ";
        for(int i=0; i<n; ++i)
            std::cout << v[i] << " ";
        std::cout << "]" << std::endl;
    }
}