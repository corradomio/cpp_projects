//
// Created by Corrado Mio on 08/03/2024.
//

#include "../exceptions.h"
#include "dot_op.h"

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

    // A.B
    matrix_t dot(const matrix_t& a, const matrix_t& b) {
        // check_dot(a, b);
        // if (a.cols() != b.rows()) throw bad_dimensions();
        size_t nr = a.rows();
        size_t nc = b.cols();
        matrix_t res(nr, nc);
        // size_t nk = a.cols();
        //
        // for(size_t i=0; i<nr; ++i) {
        //     for (size_t j=0; j<nc; ++j) {
        //         Float s=0;
        //         size_t ik=i*nk;
        //         size_t kj=j;
        //         for(size_t k=0; k<nk; ++k) {
        //             s += a[ik]*b[kj];
        //             ik += 1;
        //             kj += nc;
        //         }
        //         res[i, j] = s;
        //     }
        // }
        // return res;
        dot_eq(res, a, b, false, false);
        return res;
    }

    // A^T.B
    matrix_t tdot(const matrix_t& a, const matrix_t& b) {
        matrix_t res(a.cols(), b.cols());
        dot_eq(res, a, b, true, false);
        return res;
    }

    // A.B^T
    matrix_t dott(const matrix_t& a, const matrix_t& b) {
        matrix_t res(a.rows(), b.rows());
        dot_eq(res, a, b, false, true);
        return res;
    }

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