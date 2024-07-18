//
// Created by Corrado Mio on 19/04/2024.
//
#include <cmath>
#include <openblas/lapacke.h>
#include "stdx/float64/dot_op.h"
#include "stdx/float64/array_op.h"
#include "stdx/float64/vector_op.h"
#include "stdx/float64/matrix_op.h"
#include "stdx/float64/svd.h"

//
//  s   float
//  d   double
//  c   complex<float>
//  z   complex<double>
//
//  ge  general matrix
//
// LAPACKE: d<mtype><algo>


namespace stdx::float64 {

    // ----------------------------------------------------------------------
    // Singular Value Decomposition
    // ----------------------------------------------------------------------
    // A: n,m
    // U.D.V^T = svd(A)
    // U: n, n
    // D: n,m
    // V: m,m

    static stdx::options_t svd_opts = stdx::options_t();

    std::tuple<matrix_t, vector_t, matrix_t> svd(const matrix_t &mat, const stdx::options_t &opts) {
        matrix_t M(mat, true);
        size_t m = M.rows();
        size_t n = M.cols();

        matrix_t U(m, m);
        vector_t D(n);
        matrix_t Vt(n, n);
        vector_t superb(std::min(m, n) - 1);

        lapack_int lda = n, ldu = m, ldvt = n, info;

        // matrix_layout: LAPACK_ROW_MAJOR (C) | LAPACK_COL_MAJOR (Fortran)
        // jobu, jobvt:
        //      A)LL         All left (or right) singular vectors are returned in supplied matrix U (or Vt).
        //      C)OMPACT     The first min(m, n) singular vectors are returned in supplied matrix U (or Vt).
        //      N)O_VECTORS  No singular vectors are computed.
        //      O)VERWRITE   The first min(m, n) singular vectors are overwritten on the matrix A.

        // lapack_int LAPACKE_dgesvd( int matrix_layout, char jobu, char jobvt,
        //                            lapack_int m, lapack_int n, double* a,
        //                            lapack_int lda, double* s, double* u, lapack_int ldu,
        //                            double* vt, lapack_int ldvt, double* superb );
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A',
                              m, n,
                              M.data(), lda,
                              D.data(),
                              U.data(), ldu,
                              Vt.data(), ldvt,
                              superb.data());

        return std::make_tuple(U, D, Vt);
    }

}

