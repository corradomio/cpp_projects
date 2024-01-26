//
// Created by Corrado Mio on 10/01/2024.
//
// indices
//      forward  >= 0
//      backward < 0
//      all
//

#ifndef LINALG_H
#define LINALG_H

// (u)int(8|16|32|64)_t
// (u)int_least(8|16|32|64)_t
// (u)int_fast(8|16|32|64)_t
// (u)intmax_t
#include <stdint.h>

#include <initializer_list>
#include "refcount.h"

namespace stdx::linalg {

    static const int all = INT32_MAX;

    struct dim_t {
        static const int MAX_RANK = 5;
        int rank;
        size_t length[MAX_RANK];
        size_t stride[MAX_RANK];

        void assign(const dim_t& d) {
            rank = d.rank;
            for (int i=0; i<MAX_RANK; ++i) {
                length[i] = d.length[i];
                stride[i] = d.stride[i];
            }
        }

        explicit dim_t(const std::initializer_list<size_t>& init) {
            rank = init.size();
            int i = 0;
            for(auto it=init.begin(); it<init.end(); ++it, ++i)
                length[i] = *it;

            size_t s=1;
            for (i=rank-1; i>= 0; --i) {
                stride[i] = s;
                s *= length[i];
            }
        }

        dim_t(const dim_t& d) {
            assign(d);
        }

        dim_t& operator =(const dim_t& d) {
            assign(d);
            return *this;
        }

        size_t size() const {
            size_t s = 1;
            for (size_t i=0; i<rank; ++i)
                s *= length[i];
            return s;
        }
    };

    template<typename T>
    struct data_t: refc_t {
        size_t size;
        T data[1];

        explicit data_t(size_t sz): size(sz) { }
    };

    template<typename T>
    struct tensor {

        ref_ptr<data_t<T>> ptr;
        dim_t dim;

        void alloc() {
            void* allocated = new char[sizeof(data_t<T>) + dim.size()*sizeof(T)];
            ptr = new (allocated) data_t<T>(dim.size());
        }

        tensor(): dim({}), ptr() {
            alloc();
        }

        tensor(const std::initializer_list<size_t>& init)
        : dim(init), ptr() {
            alloc();
        }

        tensor(const tensor& t): dim(t.dim), ptr(t.ptr) {}

        tensor& operator =(const tensor& t) {
            dim = t.dim;
            ptr = t.ptr;
            return *this;
        }

        tensor operator[](int i) {
            return *this;
        }

        tensor operator[](const std::initializer_list<size_t>& init) {
            return *this;
        }
    };

}

#endif //LINALG_H
