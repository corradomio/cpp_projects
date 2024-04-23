//
// Created by Corrado Mio on 08/03/2024.
//
#include <iostream>
#include "stdx/float64/array_op.h"
#include "stdx/float64/vector_op.h"

namespace stdx::float64 {

    void check(const vector_t& u, const vector_t& v) {
        if (u.size() != v.size())
            throw bad_dimensions();
    }

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

    real_t norm(const vector_t& v, int p) {
        real_t res = 0;
        size_t n = v.size();
        switch (p){
            case 0:
                // ||v||_0
                res = reduce(static_cast<const array_t&>(v), nozero);
                break;
            case 1:
                // ||v||_1
                res = reduce(static_cast<const array_t&>(v), abs);
                break;
            case 2:
                // ||v||_2
                res = sqrt(reduce(static_cast<const array_t&>(v), sq));
                break;
            case -1:
                // ||v||_infinity
                res = abs(v[0]);
                for (size_t i=0; i<n; ++i)
                    if (abs(v[i]) > res) res = abs(v[i]);
                break;
            default:
                throw std::invalid_argument("Norm p");
        }
        return res;
    }

    real_t norm(const vector_t& u, const vector_t& v, int p) {
        check(u, v);

        real_t res = 0;
        size_t n = v.size();
        switch (p){
            case 0:
                // ||v||_0
                for(size_t i=0; i<n; ++i)
                    res += nozero(u[i]-v[i]);
                break;
            case 1:
                // ||v||_1
                for(size_t i=0; i<n; ++i)
                    res += abs(u[i]-v[i]);
                break;
            case 2:
                // ||v||_2
                for(size_t i=0; i<n; ++i)
                    res += sq(u[i]-v[i]);
                res = sqrt(res);
                break;
            case -1:
                // ||v||_infinity
                res = abs(u[0]-v[0]);
                for (size_t i=0; i<n; ++i)
                    if (abs(u[i]-v[i]) > res) res = abs(u[i]-v[i]);
                break;
            default:
                throw std::invalid_argument("Norm p");
        }
        return res;
    }

    vector_t uversor(size_t n, real_t min, real_t max) {
        vector_t v = uniform(n, min, max);
        div_eq(v, norm(v, 2));
        return v;
    }

    // ----------------------------------------------------------------------

    void linear_eq(vector_t& r, const vector_t& u, real_t s, const vector_t& v) {
        check(r, u);
        check(r, v);

        size_t n = r.size();
        for(size_t i=0; i<n; ++i)
            r[i] = u[i] + s*v[i];
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
}