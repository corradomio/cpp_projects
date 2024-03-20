//
// Created by Corrado Mio on 08/03/2024.
//
// #include <mkl.h>
#include <openblas/cblas.h>
#include "stdx/exceptions.h"
#include "stdx/float64/array_op.h"
#include "stdx/float64/dot_op.h"

namespace stdx::float64 {

    // ----------------------------------------------------------------------

    real_t dot(const vector_t& u, const vector_t& v) {
        // check_dot(u,v);
        // if (u.size() != v.size()) throw bad_dimensions();
        // size_t n = u.size();
        // real_t s = 0;
        // for (size_t i=0; i<n; ++i)
        //     s += u[i]*v[i];
        real_t s = reduce(u, mul, v, u.size(), 0, 1, 0, 1);
        return s;
    }

    vector_t dot(const matrix_t& m, const vector_t& v) {
        // check_dot(m,v);
        // if (m.cols() != v.size()) throw bad_dimensions();
        size_t nr = m.rows();
        size_t nc = m.cols();
        vector_t res(nr);

        for(size_t i=0; i<nr; ++i) {
            // real_t s = 0;
            // for (size_t j=0,ki=i*nc; j < nc; ++j,++ki) {
            //     s += m[ki] * v[j];
            // }
            // res[i] = s;
            res[i] = reduce(m, mul, v, nc, i*nc, 1, 0, 1);
        }
        return res;
    }

    vector_t dot(const vector_t& u, const matrix_t& m) {
        // check_dot(u,m);
        // if (u.size() != m.rows()) throw bad_dimensions();
        size_t nr = m.rows();
        size_t nc = m.cols();
        vector_t res(nc);

        for(size_t i=0; i<nc; ++i) {
            // real_t s = 0;
            // size_t ji=i;
            // for (size_t j=0; j < nr; ++j) {
            //     s += u[j] * m[ji];
            //     ji += nc;
            // }
            // res[i] = s;
            res[i] = reduce(u, mul, m, nr, 0, 1, i, nc);
        }
        return res;
    }

    // ----------------------------------------------------------------------
    // A.B

    void _dot_ff(matrix_t& r, const matrix_t& a, const matrix_t& b) {
        // A.B
        if (a.cols() != b.rows()) throw bad_dimensions();
        size_t nr = a.rows();
        size_t nc = b.cols();
        size_t nk = a.cols();
        if (r.rows() != nr || r.cols() != nc)  throw bad_dimensions();

        for(size_t i=0; i<nr; ++i) {
            for (size_t j=0; j<nc; ++j) {
                r[i, j] = reduce(a, mul, b, nk, i*nk, 1, j, nc);
            }
        }
    }

    void _dot_tf(matrix_t& r, const matrix_t& a, const matrix_t& b) {
        // A^T.B
        if (a.rows() != b.rows()) throw bad_dimensions();
        size_t nk = a.rows();
        size_t nr = a.cols();
        size_t nc = b.cols();
        if (r.rows() != nr or r.cols() != nc)  throw bad_dimensions();

        for(size_t i=0; i<nr; ++i) {
            for (size_t j=0; j<nc; ++j) {
                r[i, j] = reduce(a, mul, b, nk, i, nr, j, nc);
            }
        }
    }

    void _dot_ft(matrix_t& r, const matrix_t& a, const matrix_t& b) {
        // A.B^T
        if (a.cols() != b.cols()) throw bad_dimensions();
        size_t nk = a.cols();
        size_t nr = a.rows();
        size_t nc = b.rows();
        if (r.rows() != nr or r.cols() != nc)  throw bad_dimensions();

        for(size_t i=0; i<nr; ++i) {
            for (size_t j=0; j<nc; ++j) {
                r[i, j] = reduce(a, mul, b, nk, i*nk, 1, j*nk, 1);
            }
        }
    }

    void _dot_tt(matrix_t& r, const matrix_t& a, const matrix_t& b) {
        // A^T.B^T == (B.A)^T
        if (b.cols() != a.rows()) throw bad_dimensions();
        size_t nk = b.cols();
        size_t nr = b.rows();
        size_t nc = a.rows();
        if (r.cols() != nr or r.rows() != nc)  throw bad_dimensions();

        for(size_t i=0; i<nr; ++i) {
            for (size_t j=0; j<nc; ++j) {
                r[i, j] = reduce(a, mul, b, nk, i*nk, 1, j, nc);
            }
        }
    }

    void _cblas_dot(matrix_t& r, const matrix_t& a, const matrix_t& b, bool tra, bool trb) {
        cblas_dgemm(
                CBLAS_LAYOUT::CblasRowMajor,
                tra ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
                trb ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
                tra ? a.cols() : a.rows(),
                trb ? b.rows() : b.cols(),
                tra ? a.rows() : a.cols(),
                1.0,
                a.values(),
                a.cols(),
                b.values(),
                b.cols(),
                0.0,
                r.values(),
                r.cols()
        );
    }


    // R = A.B | A^T.B | A.B^T
    void dot_eq(matrix_t& r, const matrix_t& a, const matrix_t& b, bool tra, bool trb) {
        // if (!tra && !trb) {
        //     _dot_ff(r, a, b);
        // }
        // elif (tra && !trb) {
        //     _dot_tf(r, a, b);
        // }
        // elif (!tra && trb) {
        //     _dot_ft(r, a, b);
        // }
        // else {
        //     _dot_tt(r, a, b);
        // }
        _cblas_dot(r, a, b, tra, trb);
    }

    // A.B | A^T.B | A.B^T | A^T.B^T
    matrix_t dot(const matrix_t& a, const matrix_t& b, bool tra, bool trb) {
        size_t nr = tra ? a.cols() : a.rows();
        size_t nc = trb ? b.rows() : b.cols();
        matrix_t res(nr, nc);

        dot_eq(res, a, b, tra, trb);
        return res;
    }


    // ----------------------------------------------------------------------
    // R = A.diag(v) | A^T.diag(v)
    // R = diag(v).A | diag(v).A^T

    void ddot_eq(matrix_t& r, const matrix_t& m, const vector_t& v, bool tr) {

        size_t nr = m.rows();
        size_t nc = m.cols();

        if (tr) {
            if (m.rows() != v.size()) throw stdx::bad_dimensions();

            for (size_t i=0,k=0; i<nr; ++i) {
                for (size_t j=0; j<nc; ++j,++k) {
                    r[k] = v[i]*m[k];
                }
            }
        }
        else {
            if (m.cols() != v.size()) throw stdx::bad_dimensions();

            for (size_t i=0,k=0; i<nr; ++i) {
                for (size_t j=0; j<nc; ++j,++k) {
                    r[k] = v[j]*m[k];
                }
            }
        }
    }

    matrix_t ddot(const matrix_t& m, const vector_t& v, bool tr) {
        matrix_t r = matrix_t::like(m);
        ddot_eq(r, m, v, tr);
        return r;
    }

    void ddot_eq(matrix_t& r, const vector_t& v, const matrix_t& m, bool tr) {
        ddot_eq(r, m, v, !tr);
    }

    matrix_t ddot(const vector_t& v, const matrix_t& m, bool tr) {
        return ddot(m, v, !tr);
    }

    // ----------------------------------------------------------------------

    matrix_t cross(const vector_t& u, const vector_t& v) {
        size_t nr = u.size();
        size_t nc = v.size();
        matrix_t m(nr, nc);
        for(size_t i=0; i<nr; ++i)
            for(size_t j=0; j<nc; ++j)
                m[i,j] = u[i]*v[j];
        return m;
    }
}