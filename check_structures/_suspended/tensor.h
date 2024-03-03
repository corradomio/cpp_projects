//
// Created by Corrado Mio on 25/02/2024.
//

#ifndef STDX_TENSOR_H
#define STDX_TENSOR_H

# include <cstdlib>
#include <cstddef>
#include <initializer_list>
#include <exception>
#include <cstdio>
#include "../array.h"

namespace stdx {

    struct dims_t {
        static const size_t NO_DIM = size_t(-1);

        size_t  refc;
        int     rank;
        size_t  *length;
        size_t  *stride;

        // ------------------------------------------------------------------
        // utilities

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

        // ------------------------------------------------------------------
        // constructors

        dims_t(const std::initializer_list<size_t>& dims): refc(0) {
            alloc(dims.size());
            init(dims);
        }

        dims_t(const std::initializer_list<size_t>& dims, size_t n): refc(0) {
            alloc(dims.size());
            init(dims);
            normalize(n);
        }

        dims_t(const dims_t& d): refc(0) {
            alloc(d.rank);
            init(d);
        }

        ~dims_t() {
            delete[] length;
            delete[] stride;
        }

        void normalize(size_t n) {
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
                return;

            length[missing_dim] = missing_length;
            size_t s=1;
            for (i=rank-1; i>= 0; --i) {
                stride[i] = s;
                s *= length[i];
            }
        }

        // ------------------------------------------------------------------

        [[nodiscard]] bool   empty()       const { return size() == 0; }
        [[nodiscard]] size_t dim(size_t i) const { return i >= rank ? 1 : length[i]; }
        [[nodiscard]] size_t size() const {
            size_t n = 1;
            for(int i=0; i<rank; ++i)
                n *= self.length[i];
            return n;
        }
    };

    template<typename T>
    struct tensor_t {
    private:
        info_t *pinfo;
        dims_t *pdims;
        T      *pdata;

        void add_ref(bool dims_only=false) {
            if (!dims_only) pinfo->refc++;
            self.pdims->refc++;
        }
        void release(bool dims_only=false) {
            if (0 == --pdims->refc) {
                delete pdims;
            }
            if (!dims_only && 0 == --pinfo->refc) {
                delete   pinfo;
                delete[] pdata;
            }
        }

        void alloc(const std::initializer_list<size_t>& dims) {
            self.pdims = new dims_t(dims);
            alloc(self.pdims);
        }

        void alloc(dims_t* dims) {
            self.pdims = dims;
            size_t n = dims->size();
            self.pinfo = new info_t(n, n);
            self.pdata = new T[n];
            add_ref();
        }

        void init(const tensor_t& t) {
            self.pinfo = t.pinfo;
            self.pdims = t.pdims;
            self.pdata = t.pdata;
            add_ref();
        }

        void copy(const tensor_t& t) {
            size_t n = rank();
            for (int i=0; i<n; ++i) {
                self.pdims->length[i] = t.pdims->length[i];
                self.pdims->stride[i] = t.pdims->stride[i];
            }
            n = size();
            for (int i=0; i<n; ++i) {
                self.pdata[i] = t.pdata[i];
            }
        }

        void assign(const tensor_t& t) {
            t.add_ref();
            self.release();
            self.pinfo = t.pinfo;
            self.pdims = t.pdims();
            self.pdata = t.pdata;
        }

    public:
        /// empty constructor
        tensor_t(): pinfo(nullptr), pdims(nullptr), pdata(nullptr) {
            alloc({});
        }

        /// constructor with the specified dimensions
        tensor_t(const std::initializer_list<size_t>& dims): pinfo(nullptr), pdims(nullptr), pdata(nullptr) {
            alloc(dims);
        }

        /// constructor by reference or cloning
        tensor_t(const tensor_t& t, bool clone=false) {
            if (clone) {
                alloc(t.pdims);
                copy(t);
            }
            else {
                init(t);
            }
        }

        /// destructor
        ~tensor_t(){ release(); }

        // ------------------------------------------------------------------
        // operations

        /// create a fill of data, share dims
        tensor_t clone() const { return tensor_t(self, true); }

        /// create a fill of dims, share data
        tensor_t reshape(const std::initializer_list<size_t>& dims) {
            // add temporary ref to (original) dims & info
            tensor_t r(self);
            // create the (new) dims, resolve (-1) dims
            auto *ndims = new dims_t(dims, self.size());
            // release dims
            r.release(true);
            // update dims
            r.pdims = ndims;
            // add ref dims
            r.add_ref(true);
            // release ref to (original) dims & info
            // if (original) dims had a single ref, it will be destroyed
            return r;
        }

        // ------------------------------------------------------------------
        // properties

        [[nodiscard]] const dims_t& dims() const { return (*self.pdims); }
        [[nodiscard]] const info_t& info() const { return (*self.pinfo); }
        [[nodiscard]] const T*      data() const { return ( self.pdata); }

        /// tensor rank. 0 for scalars
        [[nodiscard]] size_t rank()        const { return self.dims().rank;   }
        /// tensor size (n of elements)
        [[nodiscard]] size_t size()        const { return self.dims().size(); }
        /// dimension size
        [[nodiscard]] size_t dim(size_t i) const { return self.dims().dim(i); }
        /// tensor is a scalar value
        [[nodiscard]] bool   scalar()      const { return self.dims().rank == 0;   }
        /// empty tensor
        [[nodiscard]] bool   empty()       const { return self.dims().size() == 0;   }

        // ------------------------------------------------------------------
        // assignment by reference

        tensor_t& operator =(const tensor_t& t) {
            assign(t);
            return self;
        }

        // ------------------------------------------------------------------
        // utilities

        tensor_t& dump() {
            const dims_t& dims = self.dims();
            const info_t& info = self.info();
            ::printf("dims[%d]={", dims.size());
            if (dims.rank > 0) {
                ::printf("%d", dims.dim(0));
                for (int i = 1; i < dims.rank; ++i)
                    ::printf(",%d", dims.dim(i));
            }
            ::printf("} refs=(%d, %d)\n", dims.refc, info.refc);
            return self;
        }
    };
}

#endif //STDX_TENSOR_H
