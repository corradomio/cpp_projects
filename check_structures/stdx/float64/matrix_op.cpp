//
// Created by Corrado Mio on 08/03/2024.
//
#include <stdexcept>
#include <iostream>
#include <random>
#include "../arith.h"
#include "array_op.h"
#include "matrix_op.h"

namespace stdx::float64 {

    // ----------------------------------------------------------------------

    matrix_t zeros(size_t nr, size_t nc) {
        matrix_t m{nr, nc};
        m = 0;
        return m;
    }

    matrix_t  ones(size_t nr, size_t nc) {
        matrix_t m{nr, nc};
        m = 1;
        return m;
    }

    matrix_t range(size_t nr, size_t nc) {
        matrix_t m{nr, nc};
        size_t s = nr * nc;
        for(size_t i=0; i<s; ++i)
            m[i] = 1. + i;
        return m;
    }

    matrix_t identity(size_t nr, size_t nc)  {
        if (nc == -1) nc = nr;

        matrix_t m{nr, nc};
        nr = std::min(nr, nc);
        m = 0;
        for(size_t i=0; i < nr; ++i)
            m[i,i] = 1;
        return m;
    }

    matrix_t uniform(size_t nr, size_t nc, real_t min, real_t max) {
        real_t delta = max-min;
        matrix_t m{nr, nc};
        size_t n = m.size();
        for(size_t i=0; i < n; ++i)
            m[i] = min+(::rand()*delta)/RAND_MAX;
        return m;
    }

    // ----------------------------------------------------------------------

    bool operator == (const matrix_t& a, const matrix_t& b) {
        if (a.rows() != b.rows())
            return false;
        if (a.cols() != b.cols())
            return false;

        size_t n = a.size();
        for (size_t i=0; i<n; ++i)
            if (a[i] != b[i])
                return false;

        return true;
    }

    real_t norm(const matrix_t& m, int p) {
        real_t res = 0;
        size_t n = m.size();
        switch (p){
            case 0:
                for (size_t i=0; i<n; ++i)
                    if (m[i] != 0) res += 1;
                break;
            case 1:
                // for (size_t i=0; i<n; ++i)
                //     res += abs(m[i]);
                res = reduce(abs, (array_t&)m);
                break;
            case 2:
                // for (size_t i=0; i<n; ++i)
                //     res += sq(m[i]);
                res = reduce(stdx::sq, (array_t&)m);
                res = sqrt(res);
                break;
            case -1:
                res = abs(m[0]);
                for (size_t i=0; i<n; ++i)
                    if (abs(m[i]) > res) res = abs(m[i]);
                break;
            default:
                throw std::invalid_argument("Norm p");
        }
        return res;
    }

    real_t frobenius(const matrix_t& m) {
        real_t frob = reduce(stdx::sq, (array_t&)m);
        return std::sqrt(frob);
    }

    // ----------------------------------------------------------------------

    void print(const matrix_t& m) {
        size_t nr = m.rows();
        size_t nc = m.cols();

        std::cout << "[" << std::endl;
        for(int i=0; i<nr; ++i) {
            std::cout << "  [ ";
            for(int j=0; j<nc; ++j)
                std::cout << m[i, j] << " ";
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }
}