//
// Created by Corrado Mio on 09/03/2024.
//

#ifndef STDX_FLOAT64_TRANSPOSE_H
#define STDX_FLOAT64_TRANSPOSE_H

#include "matrix.h"

namespace stdx::float64 {

    struct tr_t {
        matrix_t mat;

        explicit tr_t(const matrix_t& m): mat(m) { }
        explicit tr_t(const tr_t& that): mat(that.mat) { }
        tr_t & operator=(const tr_t& that) {
            self.mat = that.mat;
            return self;
        }

        [[nodiscard]] size_t rows() const { return mat.cols(); }
        [[nodiscard]] size_t cols() const { return mat.rows(); }

        real_t operator[](size_t i, size_t j) const {
            return mat[j, i];
        }

        real_t& operator[](size_t i) const {
            size_t n = self.rows();
            size_t m = self.cols();
            size_t r = i/m;
            size_t c = i%m;
            return mat[r*n + c];
        }

    };

    tr_t tr(const matrix_t& m);

    void print(const tr_t& tr);
}


#endif //STDX_FLOAT64_TRANSPOSE_H
