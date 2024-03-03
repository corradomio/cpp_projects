//
// Created by Corrado Mio on 02/03/2024.
//

#ifndef STDX_MATRIX_H
#define STDX_MATRIX_H

#include "array.h"

namespace stdx {

    template<typename T>
    struct matrix_t : public array_t<T> {
        size_t ncols;

        void assign(const matrix_t &that) {
            array_t<T>::assign(that);
            self.ncols = that.ncols;
        }

        matrix_t clone() const {
            return matrix_t(self, true);
        }

        // ------------------------------------------------------------------

        matrix_t() : matrix_t(0, 0) {}

        matrix_t(size_t rows, size_t cols) : array_t<T>(rows * cols), ncols(cols) {}

        matrix_t(const matrix_t &m) :matrix_t(m, false) {}

        matrix_t(const matrix_t &m, bool clone) : array_t<T>(m, clone), ncols(m.cols()) {}

        // ------------------------------------------------------------------

        matrix_t &operator=(const matrix_t &m) {
            assign(m);
            return self;
        }

        matrix_t &operator=(T s) {
            array_t<T>::fill(s);
            return self;
        }

        // ------------------------------------------------------------------

        [[nodiscard]] size_t rows() const { return self.size() / ncols; }

        [[nodiscard]] size_t cols() const { return ncols; }

    };

}

#endif //STDX_MATRIX_H
