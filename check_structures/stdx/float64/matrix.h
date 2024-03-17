//
// Created by Corrado Mio on 08/03/2024.
//

#include "array.h"

#ifndef STDX_FLOAT64_MATRIX_H
#define STDX_FLOAT64_MATRIX_H

namespace stdx::float64 {

    struct matrix_t : public array_t {
        using super = array_t;

        size_t ncols;

        // create by ref
        void init(const matrix_t& that);
        // assign vy ref
        void assign(const matrix_t& that);

        void fill(const matrix_t& that);

        // ------------------------------------------------------------------
        // Constructors

        matrix_t(): matrix_t(0,0) { };
        matrix_t(size_t n, size_t m, bool clone=false): super(n*m, clone), ncols(m) { };
        // create by ref/copy
        matrix_t(const matrix_t& that, bool clone=false): super(that, clone), ncols(that.ncols) { }
        // create by ref + reshape
        matrix_t(const matrix_t& that, size_t ncols): super(that, false), ncols(ncols) { }

        // ------------------------------------------------------------------
        // Properties

        [[nodiscard]] size_t rows() const { return self.size() / self.ncols; }
        [[nodiscard]] size_t cols() const { return self.ncols; }

        // ------------------------------------------------------------------
        // References

        [[nodiscard]] matrix_t  clone() const { return {self, true}; }
        [[nodiscard]] matrix_t norefs() const { return self.info->refc==1 ? self : self.clone(); }
        [[nodiscard]] matrix_t reshape(size_t n, size_t m);

        // ------------------------------------------------------------------
        // Assignment

        matrix_t& operator=(const matrix_t& v);
        matrix_t& operator=(real_t s);

        // ------------------------------------------------------------------
        // accessors
        // single index supported by 'array_t'

        // Float& operator[](size_t i)       { return self.data[i]; }
        real_t& operator[](size_t i) const {
            // assert(i < self.size());
            return self.data[i];
        }

        // Float& operator[](size_t i, size_t j)       { return self.data[i*self.ncols+j]; }
        real_t& operator[](size_t i, size_t j) const {
            // assert(i < self.rows());
            // assert(j < self.cols());
            return self.data[i*self.ncols+j];
        }

        // ------------------------------------------------------------------
        // Iterators

        array_it begin() { return array_it{self, 0, 1}; }
        array_it   end() { return array_it{self, self.size(), 0}; }

        array_it row_begin(size_t r) { return array_it{self, r*self.cols(), 1}; }
        array_it   row_end(size_t r) { return array_it{self, (r+1)*self.cols(), 0}; }

        array_it col_begin(size_t c) { return array_it{self,c, self.cols()}; }
        array_it   col_end(size_t c) { return array_it{self, self.rows()*self.cols()+c, 0}; }

    };

}

#endif //STDX_FLOAT64_MATRIX_H
