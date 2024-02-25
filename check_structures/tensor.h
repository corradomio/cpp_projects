//
// Created by Corrado Mio on 25/02/2024.
//

#ifndef STDX_TENSOR_H
#define STDX_TENSOR_H

#include <cstddef>
#include <initializer_list>
#include <exception>
#include "array.h"

namespace stdx {

    struct dims_t {
        static const size_t NO_DIM = size_t(-1);

        size_t  refc;
        int     rank;
        size_t *length;
        size_t *stride;

        dims_t(const std::initializer_list<size_t>& dims): refc(0) {
            alloc(dims.size());
            init(dims);
        }

        dims_t(const dims_t& d): refc(0) {
            alloc(d.rank);
            init(d);
        }

        ~dims_t() {
            delete[] length;
            delete[] stride;
        }

        void alloc(size_t r) {
            rank = r;
            length = new size_t[rank];
            stride = new size_t[rank];
        }

        void init(const std::initializer_list<size_t>& init) {
            int i = 0;
            for(auto it=init.begin(); it<init.end(); ++it, ++i)
                length[i] = *it;

            size_t s=1;
            for (i=rank-1; i>= 0; --i) {
                stride[i] = s;
                s *= length[i];
            }
        }

        void init(const dims_t& d) {
            for(int i=0; i<rank; ++i) {
                length[i] = d.length[i];
                stride[i] = d.stride[i];
            }
        }

        dims_t* normalize(size_t n) {
            size_t missing_length = n;
            int missing_dim = -1, i;
            for(i=0; i<rank; ++i) {
                if (length[i] != NO_DIM)
                    missing_length /= length[i];
                else if (missing_dim != NO_DIM)
                    // multiple missing dimensions
                    throw std::bad_array_new_length();
                else
                    missing_dim = i;
            }
            if (missing_dim == -1)
                return this;

            length[missing_dim] = missing_length;
            size_t s=1;
            for (i=rank-1; i>= 0; --i) {
                stride[i] = s;
                s *= length[i];
            }

            return this;
        }

        size_t size() const {
            size_t n = 1;
            for(int i=0; i<rank; ++i)
                n *= self.length[i];
            return n;
        }
    };

    template<typename T>
    struct tensor_t {
    private:
        info_t *info;
        dims_t *dims;
        T      *data;

        void add_ref(bool dims_only=false) {
            if (!dims_only) info->refc++;
            dims->refc++;
        }
        void release(bool dims_only=false) {
            if (0 == --dims->refc) {
                delete dims;
            }
            if (!dims_only && 0 == --info->refc) {
                delete   info;
                delete[] data;
            }
        }

        void alloc(const std::initializer_list<size_t>& dims_) {
            self.dims = new dims_t(dims_);
            alloc(self.dims);
        }

        void alloc(dims_t* dims) {
            self.dims = dims;
            size_t n = dims->size();
            self.info = new info_t(n, n);
            self.data = new T[n];
            add_ref();
        }

        void init(const tensor_t& t) {
            info = t.info;
            dims = t.dims;
            data = t.data;
            add_ref();
        }

        void copy(const tensor_t& t) {
            size_t n = rank();
            for (int i=0; i<n; ++i) {
                self.dims->length[i] = t.dims->length[i];
                self.dims->stride[i] = t.dims->stride[i];
            }
            n = size();
            for (int i=0; i<n; ++i) {
                self.data[i] = t.data[i];
            }
        }

        void assign(const tensor_t& t) {
            t.add_ref();
            self.release();
            info = t.info;
            dims = t.dims();
            data = t.data;
        }

    public:
        tensor_t()
        : info(nullptr), dims(nullptr), data(nullptr) {
            alloc({});
        }

        tensor_t(const std::initializer_list<size_t>& dims)
        : info(nullptr), dims(nullptr), data(nullptr) {
            alloc(dims);
        }

        tensor_t(const tensor_t& t, bool clone=false) {
            if (clone) {
                alloc(t.dims);
                copy(t);
            }
            else {
                init(t);
            }
        }

        ~tensor_t(){ release(); }

        tensor_t clone() const { return tensor_t(self, true); }

        [[nodiscard]] size_t rank()        const { return self.dims->rank; }
        [[nodiscard]] size_t dim(size_t i) const { return self.dims->length[i]; }
        [[nodiscard]] size_t size()        const { return self.dims->size(); }

        tensor_t reshape(const std::initializer_list<size_t>& dims) {
            dims_t *ndims = (new dims_t(dims))->normalize(self.size());
            tensor_t r(self);
            r.release(true);
            r.dims = ndims;
            r.add_ref(true);
            return r;
        }
    };
}

#endif //STDX_TENSOR_H
