//
// Created by Corrado Mio on 02/03/2024.
//

#ifndef STDX_MATRIX_H
#define STDX_MATRIX_H

#include "../array.h"
#include "vector.h"

namespace stdx::linalg {

    template<typename T> struct vector_t;
    template<typename T> struct matrix_t;

    template<typename T>
    struct matrix_t : public stdx::array_t<T> {
        size_t ncols;

        void assign(const matrix_t &that) {
            array_t<T>::assign(that);
            self.ncols = that.ncols;
        }

        // ------------------------------------------------------------------
        // constructor

        matrix_t() : matrix_t(0, 0) {}

        matrix_t(size_t rows, size_t cols) : array_t<T>(rows * cols), ncols(cols) {}

        matrix_t(const matrix_t &m) :matrix_t(m, false) {}

        matrix_t(const matrix_t &m, bool clone) : array_t<T>(m, clone), ncols(m.ncols) {}

        // ------------------------------------------------------------------
        // properties

        [[nodiscard]] size_t rows() const { return self.size() / ncols; }
        [[nodiscard]] size_t cols() const { return ncols; }

        // ------------------------------------------------------------------
        // operations

        matrix_t clone() const { return matrix_t(self, true); }

        matrix_t norefs() const { return  (self._info->refc == 1) ? self : self.clone(); }

        matrix_t reshape(size_t rows, size_t cols) const {
            // if (rows == cols && rows == size_t(-1))
            //     throw bad_dimensions();
            size_t size = self.size();
            if (rows == size_t(-1)) rows = size/cols;
            if (cols == size_t(-1)) cols = size/rows;
            if (rows*cols != size) throw bad_dimensions();

            matrix_t r(self);
            r.ncols = cols;
            return r;
        }

        // ------------------------------------------------------------------
        // accessors
        // at(i) supports negative indices

        // T& at(size_t i)       { return self._data[i]; }
        // T  at(size_t i) const { return self._data[i]; }

        // T& at(size_t i, size_t j)       { return self._data[i*self.ncols+j]; }
        // T  at(size_t i, size_t j) const { return self._data[i*self.ncols+j]; }

        T& operator[](size_t i)       { return self._data[i]; }
        T  operator[](size_t i) const { return self._data[i]; }

        T& operator[](size_t i, size_t j)       { return self._data[i*self.ncols+j]; }
        T  operator[](size_t i, size_t j) const { return self._data[i*self.ncols+j]; }

        // ------------------------------------------------------------------
        // assignment

        matrix_t &operator=(const matrix_t &m) {
            assign(m);
            return self;
        }

        matrix_t &operator=(T s) {
            self.fill(s);
            return self;
        }

        // ------------------------------------------------------------------
        // dot

        vector_t<T> dot(const vector_t<T>& v) const;
        matrix_t<T> dot(const matrix_t<T>& v) const;

    };

}

#endif //STDX_MATRIX_H
