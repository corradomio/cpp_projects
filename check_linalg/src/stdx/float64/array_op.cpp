//
// Created by Corrado Mio on 08/03/2024.
//
#include "stdx/float64/array_op.h"

namespace stdx::float64 {

    void check(const array_t& u, const array_t& v) {
        if (u.size() != v.size())
            throw bad_dimensions();
    }

    // ----------------------------------------------------------------------

    void apply_eq(array_t& u, real_t (*f)(real_t)) {
        size_t n = u.size();
        for(size_t i=0; i<n; ++i)
            u[i] = f(u[i]);
    }

    void apply_eq(array_t& u, real_t (*f)(real_t, real_t), real_t s) {
        size_t n = u.size();
        for(size_t i=0; i<n; ++i)
            u[i] = f(u[i], s);
    }

    void apply_eq(array_t& u, real_t (*f)(real_t, real_t), const array_t& v) {
        check(u,v);
        size_t n = u.size();
        for(size_t i=0; i<n; ++i)
            u[i] = f(u[i], v[i]);
    }

    void apply_eq(array_t& u, real_t (*f)(real_t, real_t, real_t), real_t s, const array_t& v) {
        check(u,v);
        size_t n = u.size();
        for(size_t i=0; i<n; ++i)
            u[i] = f(u[i], s, v[i]);
    }

    void apply_eq(array_t& u, real_t (*f)(real_t, real_t), const array_t& v, const array_t& w) {
        check(u,v);
        check(v,w);
        size_t n = u.size();
        for(size_t i=0; i<n; ++i)
            u[i] = f(v[i], w[i]);
    }

    // ----------------------------------------------------------------------

    real_t reduce(real_t (*f)(real_t), const array_t& u) {
        real_t s = 0;

        size_t n = u.size();
        for(size_t i=0; i<n; ++i)
            s += f(u[i]);
        return s;
    }

    real_t reduce(real_t (*f)(real_t, real_t), const array_t& u, real_t s) {
        real_t res = 0;

        size_t n = u.size();
        for(size_t i=0; i<n; ++i)
            res += f(u[i], s);
        return res;
    }

    real_t reduce(real_t (*f)(real_t, real_t), const array_t& u, const array_t& v,
                  size_t n,
                  size_t ou, size_t su,     // offset/skip u
                  size_t ov, size_t sv) {   // offset/skip v

        if ((ou+n*su-su) >= u.size() || (ov+n*sv-sv) >= v.size())
            throw bad_dimensions();

        real_t res = 0;
        if (ou == 0 && su == 1 && ov == 0 && sv == 1) {
            for(size_t i=0; i<n; ++i)
                res += f(u[i], v[i]);
        }
        elif (su == 1 && sv == 1) {
            for (size_t i=0,iu=ou,iv=ov; i<n; ++i,++iu,++iv)
                res += f(u[iu], v[iv]);
        }
        elif (su == 1) {
            for (size_t i=0,iu=ou,iv=ov; i<n; ++i,++iu,iv+=sv)
                res += f(u[iu], v[iv]);
        }
        elif (sv == 1) {
            for (size_t i=0,iu=ou,iv=ov; i<n; ++i,iu+=su,++iv)
                res += f(u[iu], v[iv]);
        }
        else {
            for (size_t i=0,iu=ou,iv=ov; i<n; ++i,iu+=su,iv+=sv)
                res += f(u[iu], v[iv]);
        }

        return res;
    }

    real_t reduce(real_t (*f)(real_t, real_t), const array_t& u, const array_t& v) {
        // check(u,v);
        // real_t res = 0;
        //
        // size_t n = u.size();
        // for(size_t i=0; i<n; ++i)
        //     res += f(u[i], v[i]);
        // return res;
        return reduce(f, u, v, u.size(), 0, 1, 0, 1);
    }

    // ----------------------------------------------------------------------

    real_t min(const array_t& v) {
        real_t res = v[0];

        size_t n = v.size();
        for (size_t i=0; i<n; ++i)
            if (v[i] < res) res = v[i];
        return res;
    }

    real_t max(const array_t& v) {
        real_t res = v[0];

        size_t n = v.size();
        for (size_t i=0; i<n; ++i)
            if (v[i] > res) res = v[i];
        return res;
    }
}

