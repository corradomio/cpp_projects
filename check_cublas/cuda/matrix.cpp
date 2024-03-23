//
// Created by Corrado Mio on 22/03/2024.
//
#include <iostream>
#include "cublas.h"


namespace cuda {

    // ----------------------------------------------------------------------

    void matrix_t::alloc(size_t rows, size_t cols, device_t dev) {
        super::alloc(rows*cols, dev);
        self.ncols = cols;
    }

    void matrix_t::assign(const matrix_t& that) {
        super::assign(that);
        self.ncols = that.ncols;
    }

    matrix_t& matrix_t::to(device_t dev) {
        if (dev == self.dev())
            return self;

        // the matrix in GPU must be column-wise
        if (dev == device_t::GPU)
            self.layout(layout_t::COLS);

        self.to_dev(dev);
        return self;
    }


    static real_t* inline_tr(real_t* odata, size_t nr, size_t nc) {
        // nr/nc in C style
        auto *tdata = (real_t *)malloc (nr*nc*sizeof (real_t));
        size_t h, k;
        real_t t;
        for (size_t i=0; i<nr; ++i) {
            for (size_t j=0; j<nc; ++j){
                h = i*nc + j;
                k = j*nr + i;
                tdata[k] = odata[h];
            }
        }
        free(odata);
        return tdata;
    }

    matrix_t& matrix_t::layout(layout_t lay) {
        if (self.layout() == lay)
            return self;
        // it is not possible to change the layout if the matrix is in the gpu
        if (self.dev() == device_t::GPU)
            throw cublas_error(cudaError::cudaErrorInvalidDevice);

        size_t nr = self.rows();
        size_t nc = self.cols();

        if (lay == layout_t::ROWS) {
            self._data = inline_tr(self._data, nc, nr);
        }
        else {
            self._data = inline_tr(self._data, nr, nc);
        }
        self._info->lay = lay;

        return self;
    }

    // ----------------------------------------------------------------------

    matrix_t::matrix_t() {
        self.alloc(0, 0, CPU);
        self.add_ref();
    }

    matrix_t::matrix_t(size_t rows, size_t cols, device_t dev) {
        self.alloc(rows, cols, dev);
        self.add_ref();
    }

    matrix_t::matrix_t(const matrix_t& that, bool clone) {
        if (clone) {
            self.alloc(that.rows(), that.cols(), that.dev());
            self.add_ref();
            self.fill(that);
        }
        else {
            self.assign(that);
            self.add_ref();
        }
    }

    // ----------------------------------------------------------------------

    matrix_t zeros(size_t rows, size_t cols) {
        matrix_t m{rows, cols};
        for (size_t i=0; i<rows; ++i)
            for (size_t j=0; j<cols; ++j)
                m[i*cols + j] = 0.;
        return m;
    }

    matrix_t  ones(size_t rows, size_t cols) {
        matrix_t m{rows, cols};
        for (size_t i=0; i<rows; ++i)
            for (size_t j=0; j<cols; ++j)
                m[i*cols + j] = 1.;
        return m;
    }

    matrix_t range(size_t rows, size_t cols) {
        matrix_t m{rows, cols};
        for (size_t i=0,k=0; i<rows; ++i)
            for (size_t j=0; j<cols; ++j,++k)
                m[k] = 1.+k;
        return m;
    }

    matrix_t identity(size_t rows, size_t cols) {
        if (cols == -1) cols = rows;
        matrix_t m{rows, cols};
        for (size_t i=0; i<rows; ++i)
            for (size_t j=0; j<cols; ++j)
                m[i*cols + j] = (i==j) ? 1. : 0.;
        return m;
    }

    matrix_t uniform(size_t rows, size_t cols, real_t min, real_t max) {
        matrix_t m{rows, cols};
        real_t delta = (max - min);
        size_t n = rows*cols;
        for (size_t i=0; i<n; ++i)
            m[i] = min + (delta*rand())/RAND_MAX;
        return m;
    }

    // ----------------------------------------------------------------------

    void print(const matrix_t& m) {
        size_t nr = m.rows();
        size_t nc = m.cols();

        std::cout << "{" << std::endl;
        for(int i=0; i<nr; ++i) {
            std::cout << "  { ";
            if (nc > 0)
                std::cout << m[i, 0];
            for(int j=1; j<nc; ++j)
                std::cout << ", " << m[i, j];
            std::cout << " }," << std::endl;
        }
        std::cout << "}" << std::endl;
    }

    // ----------------------------------------------------------------------

}