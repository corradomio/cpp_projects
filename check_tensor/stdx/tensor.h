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

    /**
     * Used to create a tensor and tensor/view  size
     */
    struct shape_t {
        uint16  rank;
        uint16  dims[8];

        shape_t();
        shape_t(const std::initializer_list<uint16>& dims);
        shape_t(const shape_t& shape) = default;

        [[nodiscard]] bool   empty() const { return self.size() == 0; };
        [[nodiscard]] size_t size() const;

        shape_t& operator =(const shape_t& shape) = default;
        uint16   operator[](uint16 i) const { return self.dims[i]; };
    };

    /**
     * Used to create a view inside the tensor
     */
    struct span_t {
        uint16 off;     // offset
        uint16 end;     // length
        uint16 step;    // step

        span_t(): span_t(0, ANY, 1){ }
        span_t(uint16 o): span_t(o, o+1, 1){ }
        span_t(uint16 o, uint16 e): span_t(o, e, 1){ }
        span_t(uint16 o, uint16 e, uint16 s): off(o), end(e), step(s){ }

        // [[nodiscard]] uint16 len() const { return self.end-self.off; }

        span_t(const span_t& s) = default;
        span_t& operator =(const span_t& s) = default;
    };

    /**
     * Contains the information for a single dimension
     *
     *      1) dimension size (doff, dlen, esize)
     *          doff: if the dimension 0 starts at the begin of the array or
     *                elements of the array are skipped.
     *                It is used quen the rank of the tensors is reduced because
     *                some dimensions are selected for single value
     *          dlen:  n of contiguous elements in this dimension
     *          esize: n of elements composing the dimension.
     *                It is the product of dlen of all dimensions
     *                lower than the current one
     *
     *      2) the current view in the dimension (off, len, step)
     *          off:  position of the element with index 0
     *          len:  n of elements in the view
     *          step: distance from the previous element.
     *                1 for contiguous elements
     *
     */
    struct dim_t {
        // global
        size_t doff;    // dimension offset
        uint16 dlen;    // dimension length
        size_t esize;   // dimension element size;
        // view
        uint16 off;     // view offset
        uint16 len;     // view length
        uint16 step;    // view step

        dim_t() { }
        dim_t(uint16 d): doff(0), dlen(d), esize(1), off(0), len(d), step(1) { }
        dim_t(const dim_t& that) = default;

        dim_t& operator =(const dim_t& that) = default;
    };

    /**
     * Vector rank
     */
    struct rank_t {
        uint16  r;          // n of dimensions
        dim_t   dims[8];    // properties of each dimension
        uint16  dord[8];    // dimensions order

        // rank_t(uint16 r);
        rank_t(const shape_t& dims);
        rank_t(const rank_t&  that) = default;

        /// Number of dimensions
        [[nodiscard]] uint16  rank() const;
        /// Number of elements composing the tensor (not the view)
        [[nodiscard]] size_t  size() const;
        /// Length of the selected dimensions in terms of steps
        [[nodiscard]] uint16  dim(uint16 i) const;
        /// Length of the selected dimensions in terms of dlen
        [[nodiscard]] uint16  dim_(uint16 i) const;
        /// Shape of the tensor in terms of steps and dimensions
        ///     with length not 1 (0 is included)
        [[nodiscard]] shape_t shape() const;

        void swap(uint16 d1, uint16 d2);
        void swap(const std::initializer_list<uint16>& dorder);

        /// Update the this rank to match the selected view
        void view(const std::initializer_list<span_t>& spans);
        /// Compact the rank removing dimensions with length = 1
        void compact();

        /// Location of the element at specified indices
        [[nodiscard]] size_t at(const std::initializer_list<uint16>& indices) const;
    };

    template<typename T>
    struct tensor_t {

        struct info_t {
            uint16  refc;   // refcount
            size_t  n;      // n_elements of the array

            info_t(uint16 n): refc(0), n(n){ }
        };

        // DOESN'T change the order!
        T      *_data;      // allocated memory
        info_t *_info;      // reference counting for memory
        rank_t *_rank;      // per instance tensor information

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
            return self._rank->dim_(i);
        }

        [[nodiscard]] shape_t shape() const { return self._rank->shape(); }
        [[nodiscard]] rank_t& rrank() const { return *self._rank; }

        // ------------------------------------------------------------------
        // accessors

        explicit operator float() const {
            assert(self.rank() == 0);
            return self._data[self._rank->at({})];
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
        // dimensions

        tensor_t swap(uint16 d1, uint16 d2) {
            tensor_t r(self);
            r._rank->swap(d1, d2);
            return r;
        }
        tensor_t swap(const std::initializer_list<uint16>& dorder) {
            tensor_t r(self);
            r._rank->swap(dorder);
            return r;
        }

        // ------------------------------------------------------------------
        // sub tensor

        tensor_t view(const std::initializer_list<span_t>& ranges) {
            tensor_t r(self);
            r._rank->view(ranges);
            r._rank->compact();
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
            for (uint16 i=0; i<=r; ++i) {
                dim_t& dim = self._rank->dims[i];
                printf("... [%2d] off:%3d, len:%3d, step:%3d (doff:%3d, dlen:%3d, esize:%3d)\n",
                       i,
                       dim.off,
                       dim.len,
                       dim.step,

                       dim.doff,
                       dim.dlen,
                       dim.esize
                );
            }
        }
    };

}

#endif //STDX_TENSOR_H
