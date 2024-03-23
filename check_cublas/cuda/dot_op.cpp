//
// Created by Corrado Mio on 23/03/2024.
//
#include "dot_op.h"

namespace cuda {

    real_t dot(const vector_t& u, const vector_t& v) {
        if (u.dev() != v.dev() || u.size() != v.size())
            throw cublas_error(cudaError::cudaErrorInvalidValue);

        return 0;
    }

    // ----------------------------------------------------------------------

    void dot_eq(vector_t& r, const matrix_t& m, const vector_t& v) {
        size_t nr = m.rows();
        size_t nc = m.cols();
        if (nc != v.size()) throw bad_dimensions();
        if (nr != r.size()) throw bad_dimensions();

        real_t alpha = 1;
        real_t beta = 0;
        check(cublasDgemv(context, cublasOperation_t::CUBLAS_OP_N,
                          nr, nc,
                          &alpha,
                          m.data(), nr,   // matrix column
                          v.data(), 1,
                          &beta,
                          r.data(), 1));
    }

    void dot_eq(vector_t& r, const vector_t& u, const matrix_t& m) {
        size_t nr = m.rows();
        size_t nc = m.cols();
        if (u.size() != nr) throw bad_dimensions();
        if (r.size() != nc) throw bad_dimensions();

        real_t alpha = 1;
        real_t beta  = 0;
        check(cublasDgemv(context, cublasOperation_t::CUBLAS_OP_T,
                         nr, nc,
                         &alpha,
                         m.data(), nr,
                         u.data(), 1,
                         &beta,
                         r.data(), 1));
    }

    void dot_eq(matrix_t& r, const matrix_t& a, const matrix_t& b, bool tra, bool trb) {
        real_t alpha = 1;
        real_t beta  = 0;
        check(cublasDgemm(context,
                         tra ? cublasOperation_t::CUBLAS_OP_T :  cublasOperation_t::CUBLAS_OP_N,
                         trb ? cublasOperation_t::CUBLAS_OP_T :  cublasOperation_t::CUBLAS_OP_N,
                         a.rows(), b.cols(), a.cols(),
                         &alpha,
                         a.data(), a.rows(),
                         b.data(), b.rows(),
                         &beta,
                         r.data(), r.rows()));

    }

    void cross_eq(matrix_t& m, const vector_t& u, const vector_t& v) {

    }

    // ----------------------------------------------------------------------

    vector_t dot(matrix_t& m, const vector_t& v) {
        vector_t r{m.rows(), m.dev()};
        dot_eq(r, m, v);
        return r;
    }

    vector_t dot(const vector_t& u, const matrix_t& m) {
        vector_t r{m.cols(), m.dev()};
        dot_eq(r, u, m);
        return r;
    }

    matrix_t cross(const vector_t& u, const vector_t& v) {
        matrix_t r{u.size(), v.size()};
        cross_eq(r, u, v);
        return r;
    }

    matrix_t dot(const matrix_t& a, const matrix_t& b, bool tra, bool trb) {
        size_t nr = tra ? a.cols() : a.rows();
        size_t nc = trb ? b.rows() : b.cols();
        matrix_t r{nr, nc, a.dev()};
        dot_eq(r, a, b, tra, trb);
        return r;
    }

    // ----------------------------------------------------------------------

}