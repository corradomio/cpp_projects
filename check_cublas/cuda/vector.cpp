//
// Created by Corrado Mio on 22/03/2024.
//
#include <iostream>
#include "cublas.h"

namespace cuda {

    // ----------------------------------------------------------------------

    vector_t::vector_t() {
        self.alloc(0, CPU);
        self.add_ref();
    }

    vector_t::vector_t(size_t n, device_t dev) {
        self.alloc(n, dev);
        self.add_ref();
    }

    vector_t::vector_t(const vector_t& that, bool clone) {
        if (clone) {
            self.alloc(that.size(), that.dev());
            self.add_ref();
            self.fill(that);
        }
        else {
            self.assign(that);
            self.add_ref();
        }
    }

    // ----------------------------------------------------------------------

    vector_t zeros(size_t n) {
        vector_t v{n};
        for (size_t i=0; i<n; ++i)
            v[i] = 0.;
        return v;
    }

    vector_t  ones(size_t n) {
        vector_t v{n};
        for (size_t i=0; i<n; ++i)
            v[i] = 1.;
        return v;
    }

    vector_t range(size_t n) {
        vector_t v{n};
        for (size_t i=0; i<n; ++i)
            v[i] = 1.+i;
        return v;
    }

    vector_t uniform(size_t n, real_t min, real_t max) {
        vector_t v{n};
        real_t delta = (max - min);
        for (size_t i=0; i<n; ++i)
            v[i] = min + (delta*rand())/RAND_MAX;
        return v;
    }

    // ----------------------------------------------------------------------

    void print(const vector_t& v) {
        size_t n = v.size();
        std::cout << "{ ";
        if (n>0)
            std::cout << v[0];
        for(int i=1; i<n; ++i)
            std::cout << ", " << v[i];
        std::cout << " }" << std::endl;
    }
};