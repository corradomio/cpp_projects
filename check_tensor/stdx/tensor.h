//
// Created by Corrado Mio on 10/06/2024.
//

#ifndef STDX_TENSOR_H
#define STDX_TENSOR_H

#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <initializer_list>
#include "language.h"
#include "intfloat.h"

/*
 * rank 0   scalar
 * rank 1   vector
 * rank 2   matrix
 */

// dimension size
typedef int uint16;


namespace stdx::linalg {

    constexpr uint16 ANY = uint16(-1);

    struct shape_t {
        uint16  rank;
        uint16  dims[8];

        shape_t(): shape_t(0){ }
        shape_t(uint16 r);
        shape_t(const std::initializer_list<uint16>& dims);
        shape_t(const shape_t& shape) = default;

        shape_t& operator =(const shape_t& shape) = default;

        [[nodiscard]] bool   empty() const { return self.size() == 0; };
        [[nodiscard]] size_t size() const;
        uint16 operator[](uint16 i) const { return self.dims[i]; };
    };

    // span:
    //  span()      == span(0,ALL,1)
    //  span(b)     == span(b,b+1,1)
    //  span(b,e)   == span(b,e,1)

    struct span_t {
        uint16 off;     // offset
        uint16 end;     // length
        int    step;    // step

        span_t():                          span_t(0, ANY, 1) { }
        span_t(uint16 o):                  span_t(o, o+1, 1) { }
        span_t(uint16 o, uint16 e):        span_t(o, e, (o <= e) ? 1 : -1) { }
        span_t(uint16 o, uint16 e, int s): off(o), end(e), step(s) {
            assert (o == ANY || e == ANY || o <= e);
        }

        span_t(const span_t& s) = default;
        span_t& operator =(const span_t& s) = default;

        [[nodiscard]] uint16 len() const { return self.end-self.off; }
    };

    struct dim_t {
        // global
        uint16 dlen;    // dimension length (
        size_t esize;   // dimension element size;
        // view
        uint16 off;     // offset
        uint16 len;     // length
        uint16 step;    // step

        dim_t() { }
        dim_t(uint16 d): dlen(d), esize(1), off(0), len(d), step(1) { }
        dim_t(const dim_t& that) = default;
        dim_t& operator =(const dim_t& that) = default;

        // dimension in terms of steps
        [[nodiscard]] uint16 dim_() const { return self.len/self.step; }
    };

    struct rank_t {
        uint16  r;   // n_dimensions
        dim_t   dims[8];

        rank_t(uint16 r);
        rank_t(const shape_t& dims);
        rank_t(const rank_t& that) = default;

        // -- based on dims[i].len/dims[i].step
        [[nodiscard]] uint16  rank() const;
        [[nodiscard]] size_t  size() const;
        [[nodiscard]] uint16  dim(uint16 i) const;
        [[nodiscard]] shape_t shape() const;

        // -- based on dims[i].dim
        [[nodiscard]] size_t  size_() const;

        [[nodiscard]] size_t at(const std::initializer_list<uint16>& indices) const;
        void sel(const std::initializer_list<span_t>& spans);
    };

    template<typename T>
    struct tensor_t {

        struct info_t {
            uint16  refc;   // refcount
            size_t  n;      // n_elements of the array

            info_t(uint16 n): refc(0), n(n){ }
        };

        // DOESN'T change the order!
        T      *_data;      // reference counted
        info_t *_info;      // reference counted
        rank_t *_rank;      // instance

        void add_ref() const { self._info->refc++; }

        void release() { if (0 == --self._info->refc) {
            delete[] self._data;
            delete   self._info;
        }}

    private:

        void alloc(const shape_t& shape) {
            // init by allocation
            size_t n = shape.size();
            self._info = new info_t(n);
            self._rank = new rank_t(shape);
            self._data = new T[n+1];
            self.add_ref();
        }

        void init(const tensor_t &that) {
            // init by refcount
            self._info = that._info;
            self._rank = new rank_t(that.rrank());
            self._data = that._data;
            self.add_ref();
        }

        void assign(const tensor_t &that) {
            // assign by refcount
            that.add_ref();
            self.release();
            self._info = that._info;
            self._rank = new rank_t(that.rrank());
            self._data = that._data;
        }

        void fill(const T &s) {
            // init _data with s
            size_t n = self._info->n;

            // NOTE: there at minimum 1 element (the scalar)
            for (uint16 i = 0; i <= n; ++i) self._data[i] = s;
        }

        void copy(const tensor_t &that) {
            // init _data with the content of that
            if (self._info->n != that._info->n)
                throw std::runtime_error("Tensor size mismatch");

            // NOTE: there at minimum 1 element (the scalar)
            size_t n = self._info->n;
            memcpy(self._data, that._data, (n+1) * sizeof(T));
        }

    public:
        // ------------------------------------------------------------------
        // constructors
        tensor_t(): tensor_t(shape_t()) { }

        tensor_t(const std::initializer_list<uint16>& dims): tensor_t(shape_t(dims)) { }

        tensor_t(const shape_t& shape) { alloc(shape); }

        tensor_t(const tensor_t& that, bool clone=false) {
            if (clone) {
                alloc(that.shape());
                copy(that);
            }
            else {
                init(that);
            }
        }

        ~tensor_t() {
            release();
            delete self._rank;
        }

        // ------------------------------------------------------------------
        // properties

        [[nodiscard]] size_t size() const { return self._info->n; }
        [[nodiscard]] bool  empty() const { return self._info->n == 0; }
        [[nodiscard]] T*     data() const { return self._data; }

        [[nodiscard]] uint16 rank() const { return self._rank->r; }
        [[nodiscard]] uint16 dim(uint16 i) const {
            assert (i <= self._rank->r);
            return self._rank->dims[i].dim_();
        }

        [[nodiscard]] shape_t shape() const { return self._rank->shape(); }
        [[nodiscard]] rank_t& rrank() const { return *self._rank; }

        // ------------------------------------------------------------------
        // accessors

        explicit operator float() const {
            assert(self.rank() == 0);
            return self._data[0];
        }

        float  at(const std::initializer_list<uint16>& indices) const {
            return self._data[self._rank->at(indices)];
        }
        float& at(const std::initializer_list<uint16>& indices) {
            return self._data[self._rank->at(indices)];
        }

        float operator [](const std::initializer_list<uint16>& indices) const {
            return self._data[self._rank->at(indices)];
        }

        float& operator [](const std::initializer_list<uint16>& indices) {
            return self._data[self._rank->at(indices)];
        }

        // ------------------------------------------------------------------
        // sub tensor

        tensor_t sel(const std::initializer_list<span_t>& ranges) {
            tensor_t r(self);
            r._rank->sel(ranges);
            return r;
        }

        // ------------------------------------------------------------------
        // assignment

        tensor_t &operator=(const T &s) {
            self.fill(s);
            return self;
        }

        tensor_t &operator=(const tensor_t &that) {
            if (this == &that) {} // disable warning
            self.assign(that);
            return self;
        }

        void dump() {
            uint16 r = self._rank->r;
            printf("rank: %d\n", r);
            for (uint16 i=0; i<r; ++i) {
                printf("... [%2d] %3d, %3d, %3d\n",
                       i,
                       self._rank->dims[i].off,
                       self._rank->dims[i].len,
                       self._rank->dims[i].step);
            }
        }
    };

}

#endif //STDX_TENSOR_H
