//
// Created by Corrado Mio on 08/03/2024.
//
#include "matrix.h"

namespace stdx::float64 {

    // ----------------------------------------------------------------------

    void matrix_t::init(const matrix_t& that) {
        array_t::init(that);
        self.ncols = that.ncols;
    }

    void matrix_t::assign(const matrix_t& that) {
        array_t::assign(that);
        self.ncols = that.ncols;
    }

    void matrix_t::fill(const matrix_t& that) {
        array_t::fill(that);
        self.ncols = that.ncols;
    }

    matrix_t matrix_t::reshape(size_t n, size_t m) {
        if (n*m != self.size())
            throw bad_dimensions();
        return {self, m};
    }

    // ----------------------------------------------------------------------

    matrix_t& matrix_t::operator=(const matrix_t& that) {
        if (this == &that) {}
        assign(that);
        return self;
    }

    matrix_t& matrix_t::operator=(real_t s) {
        array_t::fill(s);
        return self;
    }

    // ----------------------------------------------------------------------

}