//
// Created by Corrado Mio on 30/03/2023.
//

#ifndef CHECK_PARALLEL_KERNEL_KERNEL_H
#define CHECK_PARALLEL_KERNEL_KERNEL_H

#include <stddef.h>

struct dim {
    size_t x, y, z;

    dim(size_t x_=1, size_t y_=1, size_t z_=1): x(x_),y(y_),z(z_) {}
    dim(const dim& d): x(d.x), y(d.y), z(d.z) {}

    struct iterator {
        const dim& d;
        size_t x, y, z;
        iterator(const dim& d_): d(d_), x(d.x), y(d.y), z(d.z) {}
    };
};

struct range {
    const size_t s;
    const size_t e;
    const size_t d;

    struct iterator {
        const range& r;
        size_t at;
        iterator(const range& r_): r(r_), at(r.s){}
        operator size_t() const { return at; }
        size_t operator ++(int) {
            size_t it = at;
            at += r.d;
            return it;
        }
        size_t operator ++() { return at += r.d; }
    };

    range(size_t e_=1): s(0), e(e_), d(1) {}
    range(size_t s_, size_t e_, size_t d_=1): s(s_), e(e_), d(d_) {}

    iterator begin() const { return iterator(*this); }

    size_t end() const { return e; }
};

struct kernel {

    dim clusterDim;
    dim gridDim;
    dim blockDim;

    kernel(dim blockDim_): blockDim(blockDim_) {}
    kernel(dim gridDim_, dim blockDim_): gridDim(gridDim_), blockDim(blockDim_) {}
    kernel(dim clusterDim_, dim gridDim_, dim blockDim_)
        : clusterDim(clusterDim_)
        , gridDim(gridDim_)
        , blockDim(blockDim_) {}

    void evaluate() {

    }
};

#endif //CHECK_PARALLEL_KERNEL_KERNEL_H
