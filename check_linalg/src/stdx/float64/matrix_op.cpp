//
// Created by Corrado Mio on 08/03/2024.
//
#include <stdexcept>
#include <iostream>
#include <random>
#include "stdx/exceptions.h"
#include "stdx/float64/arith.h"
#include "stdx/float64/array_op.h"
#include "stdx/float64/matrix_op.h"

namespace stdx::float64 {

    void check(const matrix_t& a, const matrix_t& b) {
        if (a.rows() != b.rows() || a.cols() != b.cols())
            throw bad_dimensions();
    }

    void check(const matrix_t& a, const matrix_t& b, const matrix_t& c) {
        if (a.rows() != b.rows() || a.cols() != b.cols())
            throw bad_dimensions();
        if (a.rows() != c.rows() || a.cols() != c.cols())
            throw bad_dimensions();
    }

    // ----------------------------------------------------------------------

    matrix_t zeros(size_t nr, size_t nc) {
        matrix_t m{nr, nc};
        m = 0;
        return m;
    }

    matrix_t zeros_like(const matrix_t& m) {
        return zeros(m.rows(), m.cols());
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

    bool equals (const matrix_t& a, const matrix_t& b, real_t eps) {
        if (a.rows() != b.rows())
            return false;
        if (a.cols() != b.cols())
            return false;

        size_t n = a.size();
        for (size_t i=0; i<n; ++i) {
            real_t diff = a[i] - b[i];
            if (diff < -eps || eps < diff)
                return false;
        }

        return true;
    }

    // ----------------------------------------------------------------------

    real_t min(const matrix_t& m) {
        return min(static_cast<const array_t&>(m));
    }

    real_t max(const matrix_t& m) {
        return max(static_cast<const array_t&>(m));
    }

    // ----------------------------------------------------------------------

    real_t norm(const matrix_t& m, int p) {
        real_t res = 0;
        size_t n = m.size();
        switch (p){
            case 0:
                // ||v||_0
                res = reduce(nozero,static_cast<const array_t&>(m));
                break;
            case 1:
                // ||v||_1
                res = reduce(abs, static_cast<const array_t&>(m));
                break;
            case 2:
                // ||v||_2
                res = sqrt(reduce(sq, static_cast<const array_t&>(m)));
                break;
            case -1:
                // ||v||_infinity
                res = abs(m[0]);
                for (size_t i=0; i<n; ++i)
                    if (abs(m[i]) > res) res = abs(m[i]);
                break;
            default:
                throw std::invalid_argument("Norm p");
        }
        return res;
    }

    // ----------------------------------------------------------------------

    real_t frobenius(const matrix_t& m) {
        real_t frob = reduce(sq, (array_t&)m);
        return std::sqrt(frob);
    }

    real_t frobenius(const matrix_t& a, const matrix_t& b) {
        real_t frob = reduce(sqsub, (array_t&)a, b);
        return std::sqrt(frob);
    }

    // ----------------------------------------------------------------------

    void neg_eq(matrix_t& r) {
        apply_eq(static_cast<array_t&>(r), neg);
    }

    void sum_eq(matrix_t& r, const matrix_t& a, const matrix_t& b) {
        check(r, a, b);
        apply_eq(static_cast<array_t&>(r),
                 sum, static_cast<const array_t&>(a), static_cast<const array_t&>(b));
    }

    void sub_eq(matrix_t& r, const matrix_t& a, const matrix_t& b) {
        check(r, a, b);
        apply_eq(static_cast<array_t&>(r),
                 sub, static_cast<const array_t&>(a), static_cast<const array_t&>(b));
    }

    void mul_eq(matrix_t& r, const matrix_t& a, const matrix_t& b) {
        check(r, a, b);
        apply_eq(static_cast<array_t&>(r),
                 mul, static_cast<const array_t&>(a), static_cast<const array_t&>(b));
    }

    void div_eq(matrix_t& r, const matrix_t& a, const matrix_t& b) {
        check(r, a, b);
        apply_eq(static_cast<array_t&>(r),
                 div, static_cast<const array_t&>(a), static_cast<const array_t&>(b));
    }

    // ----------------------------------------------------------------------

    matrix_t sum(const matrix_t& a, const matrix_t& b) {
        matrix_t r = like(a);
        sum_eq(r, a, b);
        return r;
    }

    matrix_t sub(const matrix_t& a, const matrix_t& b) {
        matrix_t r = like(a);
        sub_eq(r, a, b);
        return r;
    }

    matrix_t mul(const matrix_t& a, const matrix_t& b) {
        matrix_t r = like(a);
        mul_eq(r, a, b);
        return r;
    }

    matrix_t div(const matrix_t& a, const matrix_t& b) {
        matrix_t r = like(a);
        div_eq(r, a, b);
        return r;
    }

    // ----------------------------------------------------------------------

    matrix_t tr(const matrix_t& m) {
        size_t nr = m.rows();
        size_t nc = m.cols();
        matrix_t r(nc, nr);

        size_t ri;
        size_t mi = 0;
        for (size_t i=0; i<nr; ++i) {
            ri = i;
            for (size_t j = 0; j < nc; ++j) {
                // r[j, i] = m[i, j];
                r[ri] = m[mi];
                ri += nr;
                mi += 1;
            }
        }

        return r;
    }

    // ----------------------------------------------------------------------

    real_t chopf(real_t x, real_t eps) {
        return (-eps <= x && x <= eps) ? 0. : x;
    }

    matrix_t chop(const matrix_t& m, real_t eps) {
        matrix_t r(m, true);
        apply_eq(r, chopf, eps);
        return r;
    }

    // -----------------------------

    void print(const matrix_t& m, array_style style) {
        size_t nr = m.rows();
        size_t nc = m.cols();

        const char* lpar = "<";
        const char* rpar = ">";

        switch (style) {
            case PYTHON:
                lpar="[";
                rpar="]";
                break;
            case MATHEMATICA:
                lpar="{";
                rpar="}";
                break;
        }

        // std::cout << lpar << std::endl;
        printf("%s\n", lpar);
        
        for(int i=0; i<nr; ++i) {
            // std::cout << "  " << lpar;
            printf("  %s", lpar);
            if (nc > 0)
                // std::cout << m[i, 0];
                printf("%f", m[i, 0]);
            for(int j=1; j<nc; ++j)
                // std::cout << ", " << m[i, j];
                printf(", %f", m[i, j]);
            
            if (i < (nr-1))
                // std::cout << " " << rpar << "," << std::endl;
                printf(" %s,\n", rpar);
            else
                // std::cout << " " << rpar << std::endl;
                printf(" %s\n", rpar);
        }
        // std::cout << rpar << std::endl;
        printf("%s\n", rpar);
    }

    void print_dim(const char* name, const matrix_t& m) {
        std::cout << name << ": (" << m.rows() << "," << m.cols() << ")" << std::endl;
    }

    // ----------------------------------------------------------------------

}