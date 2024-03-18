//
// Created by Corrado Mio on 08/03/2024.
//

#include "stdx/exceptions.h"
#include "stdx/float64/dot_op.h"

namespace stdx::float64 {

    // ----------------------------------------------------------------------

    real_t dot(const vector_t& u, const vector_t& v) {
        // check_dot(u,v);
        if (u.size() != v.size()) throw bad_dimensions();
        size_t n = u.size();
        real_t s = 0;
        for (size_t i=0; i<n; ++i)
            s += u[i]*v[i];
        return s;
    }

    vector_t dot(const matrix_t& m, const vector_t& v) {
        // check_dot(m,v);
        if (m.cols() != v.size()) throw bad_dimensions();
        size_t nr = m.rows();
        size_t nc = m.cols();
        vector_t res(nr);

        for(size_t i=0; i<nr; ++i) {
            real_t s = 0;
            for (size_t j=0,ki=i*nc; j < nc; ++j,++ki) {
                s += m[ki] * v[j];
            }
            res[i] = s;
        }
        return res;
    }

    vector_t dot(const vector_t& u, const matrix_t& m) {
        // check_dot(u,m);
        if (u.size() != m.rows()) throw bad_dimensions();
        size_t nr = m.rows();
        size_t nc = m.cols();
        vector_t res(nc);

        for(size_t i=0; i<nc; ++i) {
            real_t s = 0;
            size_t ji=i;
            for (size_t j=0; j < nr; ++j) {
                s += u[j] * m[ji];
                ji += nc;
            }
            res[i] = s;
        }
        return res;
    }

    // ----------------------------------------------------------------------

    // R = A.B | A^T.B | A.B^T
    void dot_eq(matrix_t& r, const matrix_t& a, const matrix_t& b, bool tra, bool trb) {
        if (!tra && !trb) {
            // A.B
            if (a.cols() != b.rows()) throw bad_dimensions();
            size_t nr = a.rows();
            size_t nc = b.cols();
            size_t nk = a.cols();
            if (r.rows() != nr or r.cols() != nc)  throw bad_dimensions();

            for(size_t i=0; i<nr; ++i) {
                for (size_t j=0; j<nc; ++j) {
                    real_t s=0;
                    size_t ik=i*nk;
                    size_t kj=j;
                    for(size_t k=0; k<nk; ++k) {
                        s += a[ik]*b[kj];
                        ik += 1;
                        kj += nc;
                    }
                    r[i, j] = s;
                }
            }
        }
        elif (tra && ! trb) {
            // A^T.B
            if (a.rows() != b.rows()) throw bad_dimensions();
            size_t nk = a.rows();
            size_t nr = a.cols();
            size_t nc = b.cols();
            if (r.rows() != nr or r.cols() != nc)  throw bad_dimensions();

            for(size_t i=0; i<nr; ++i) {
                for (size_t j=0; j<nc; ++j) {
                    real_t s=0;
                    size_t ik=i;
                    size_t kj=j;
                    for(size_t k=0; k<nk; ++k) {
                        s += a[ik]*b[kj];
                        ik += nr;
                        kj += nc;
                    }
                    r[i,j] = s;
                }
            }
        }
        elif (!tra && true) {
            // A.B^T
            if (a.cols() != b.cols()) throw bad_dimensions();
            size_t nk = a.cols();
            size_t nr = a.rows();
            size_t nc = b.rows();
            if (r.rows() != nr or r.cols() != nc)  throw bad_dimensions();

            for(size_t i=0; i<nr; ++i) {
                for (size_t j=0; j<nc; ++j) {
                    real_t s=0;
                    size_t ik=i*nk;
                    size_t kj=j*nk;
                    for(size_t k=0; k<nk; ++k) {
                        s += a[ik]*b[kj];
                        ik += 1;
                        kj += 1;
                    }
                    r[i,j] = s;
                }
            }
        }
    }

    void dot_eq(matrix_t& r, const matrix_t& a, const matrix_t& b, const matrix_t& c, bool tra, bool trb, bool trc) {
        matrix_t t = dot(b, c, trb, trc);
        dot_eq(r, a, t, tra, false);
    }

    // A.B | A^T.B | A.B^T
    matrix_t dot(const matrix_t& a, const matrix_t& b, bool tra, bool trb) {
        size_t nr = tra ? a.cols() : a.rows();
        size_t nc = trb ? b.rows() : b.cols();
        matrix_t res(nr, nc);

        dot_eq(res, a, b, tra, trb);
        return res;
    }

    matrix_t dot(const matrix_t& a, const matrix_t& b, const matrix_t& c, bool tra, bool trb, bool trc) {
        size_t nr = tra ? a.cols() : a.rows();
        size_t nc = trc ? c.rows() : c.cols();
        matrix_t res(nr, nc);

        dot_eq(res, a, b, c, tra, trb, trc);
        return res;
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