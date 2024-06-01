//
// Created by Corrado Mio on 19/04/2024.
//
#include <cmath>
#include "stdx/float64/dot_op.h"
#include "stdx/float64/array_op.h"
#include "stdx/float64/vector_op.h"
#include "stdx/float64/matrix_op.h"
#include "stdx/float64/svd.h"


namespace stdx::float64 {

    static stdx::options_t svd_opts = stdx::options_t()
            ("eps", 1.e-8)
            ("niter", 1000)
            ("method", "power")
            ("positive", false)
            ("verbose", 0)
    ;

    std::pair<real_t, vector_t> power_method(const matrix_t& m, const stdx::options_t& opts) {
        auto eps   = opts.get<real_t>("eps", svd_opts);
        auto niter = opts.get<size_t>("niter", svd_opts);
        auto verbose = opts.get<size_t>("verbose", svd_opts);
        if (niter == 0) niter = 0x0FFFFFFFFFFFFFFFLL;

        size_t it = 0;
        vector_t u = uversor(m.cols(), -1., +1);
        vector_t v = uversor(m.cols(), -1., +1);

        real_t   eval = 0;
        vector_t evec;

        while(norm(u, v) > eps && it < niter) {
            dot_eq(v, m, u);
            eval = norm(v);
            div_eq(v, eval);
            evec = v.clone();

            swap(u, v);
        }

        // return std::make_pair(eval, evec);
        return std::make_pair(eval, u);
    }

    std::pair<real_t, vector_t> lanczos_method(const matrix_t& m, const stdx::options_t& opts) {
        auto eps   = opts.get<real_t>("eps", svd_opts);
        auto niter = opts.get<size_t>("niter", svd_opts);
        auto verbose = opts.get<size_t>("verbose", svd_opts);
        if (niter == 0) niter = 0x0FFFFFFFFFFFFFFFLL;

        size_t it = 0;
        real_t alpha, beta;
        vector_t r(m.cols());
        vector_t v = uversor(m.cols(), -1., +1);
        vector_t u = dot(m, v);
        vector_t w = div(u, norm(u));

        vector_t evec;

        while(norm(w, v) > eps && it < niter) {
            alpha = dot(u, v);
            linear_eq(r, u, -alpha, v);
            beta = norm(r, 2);
            div_eq(v, beta);
            dot_eq(u, m, v);
            linear_eq(u, u, -beta, v);
        }

        return std::make_pair(beta, v);
    }

    std::pair<real_t, vector_t> largest_eigenval(const matrix_t& m, const stdx::options_t& opts) {
        if (m.rows() != m.cols())
            throw bad_dimensions("not squared");

        std::pair<real_t, vector_t> ret;
        auto method = opts.get<std::string>("method", svd_opts);
        if ("power" == method)
            ret = power_method(m, opts);
        elif ("lanczos" == method)
            ret = lanczos_method(m, opts);
        else
            throw stdx::unsupported_method(method);

        if (opts.get<bool>("positive", svd_opts) && ret.second[0] < 0) {
            ret.second = -(ret.second);
        }

        return ret;
    }

}

