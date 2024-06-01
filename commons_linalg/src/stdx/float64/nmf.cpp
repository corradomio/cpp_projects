//
// Created by Corrado Mio on 17/03/2024.
//
#include <cmath>
#include "stdx/float64/dot_op.h"
#include "stdx/float64/array_op.h"
#include "stdx/float64/matrix_op.h"
#include "stdx/float64/nmf.h"

namespace stdx::float64 {

    // ----------------------------------------------------------------------
    // Non negative matrix factorization
    // ----------------------------------------------------------------------
    // https://en.wikipedia.org/wiki/Non-negative_matrix_factorization

    static stdx::options_t nmf_opts = stdx::options_t()
            ("eps", 1.e-8)
            ("niter", 1000)
            ("verbose", 0);

    std::tuple<matrix_t, matrix_t> nmf(const matrix_t& V, size_t k, const stdx::options_t& opts) {
        // V       =   W      .  H
        // (10, 5) -> (10, 3) . (3, 5)
        //
        // WtV:   (3,5)
        // WtWH:  (3,5)
        // WQ:    (3,5)     WtV/WtWH
        //
        // VHt:  (10,3)
        // WHHt: (10,3)
        // HQ:   (10,3)     VHt/WHHt
        //
        // WH:   (10,5)
        //
        auto eps   = opts.get<real_t>("eps", nmf_opts);
        auto niter = opts.get<size_t>("niter", nmf_opts);
        auto verbose = opts.get<size_t>("verbose", nmf_opts);
        if (niter == 0) niter = 0x0FFFFFFFFFFFFFFFLL;

        size_t it = 0;
        real_t min_ = std::sqrt(min(V));
        real_t max_ = std::sqrt(max(V));

        // V[nr,nc] ~= W[nr, k] . H[k, nc]
        matrix_t W = uniform(V.rows(), k, min_, max_);
        matrix_t H = uniform(k, V.cols(), min_, max_);

        // temporary
        matrix_t WtV  = like(H);
        matrix_t WtWH = like(H);
        matrix_t WQ   = like(H);

        matrix_t VHt  = like(W);
        matrix_t WHHt = like(W);
        matrix_t HQ   = like(W);

        matrix_t WH   = like(V);

        // errors
        real_t err = frobenius(V - dot(W,H));
        real_t pre = err + 2 * eps*err;

        while (abs(pre - err) > eps && it < niter) {

            // H' = H * W^T.V / W^T.W.H
            dot_eq(WtV, W, V, true, false);
            dot_eq(WH, W, H, false, false);
            dot_eq(WtWH, W, WH, true, false);
            div_eq(WQ, WtV, WtWH);
            mul_eq(H, H, WQ);

            // W' = W * V.H^T / W.H.H^T
            dot_eq(VHt, V, H, false, true);
            dot_eq(WH, W, H, false, false);
            dot_eq(WHHt, WH, H, false, true);
            div_eq(HQ, VHt, WHHt);
            mul_eq(W, W, HQ);

            pre = err;
            it += 1;

            // err = frobenius(V - W.H)
            dot_eq(WH, W, H);
            err = frobenius(V, WH);

            if (verbose && it%verbose == 0)
                printf("[%5zu] %.5g\n", it, err);
        }
        if (verbose)
            printf("[%5zu] %.5g\n", it, err);

        return {W, H};
    }

}
