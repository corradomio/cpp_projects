//
// Created by Corrado Mio on 03/06/2024.
//

#ifndef CUDA_CUDAMEM_H
#define CUDA_CUDAMEM_H

#include <cstdlib>
#include <stdexcept>
#include <cuda.h>
#include "language.h"

#define MIN_ALLOC 16

namespace cudacpp {

    enum loc_t {
        host, device, host_locked,
        host_mapped, device_mapped,
        unified
    };

    void* cuda_alloc(size_t n, size_t esize, loc_t loc);
    void* cuda_free(void* t, loc_t loc);
    void* cuda_copy(void* dst, loc_t dst_loc, void* src, loc_t src_loc, size_t size);

    namespace detail {

        struct info_t {
            size_t refc;    // refcount
            size_t n;       // size
            size_t c;       // capacity
            loc_t  loc;     // location
            void*  ptr;     // saved host pointer for pinned memory

            info_t(size_t n, loc_t l) : refc(0), c(n), n(n), loc(l) {}
            info_t(size_t c, size_t n, loc_t l) : refc(0), c(c), n(n), loc(l) {}
        };
    }

    template<typename T>
    struct array_t {
        // DOESN'T change the order!
        T *_data;
        detail::info_t *_info;

        void add_ref() const { self._info->refc++; }
        void release() { if (0 == --self._info->refc) {
            if (self._info->loc == device_mapped) {
                self._data = reinterpret_cast<T*>(self._info->ptr);
                self._info->loc = host_mapped;
            }
            cuda_free(self._data, self._info->loc);
            delete   self._info;
        }}

        void alloc(size_t c, size_t n, loc_t loc) {
            n = (n == -1) ? c : n;

            // c must be multiple than MIN_ALLOC
            size_t d = c % MIN_ALLOC;
            c = c + (d ? (MIN_ALLOC - d) : 0);

            self._info = new detail::info_t(c, n, loc);
            self._data = (T*)cuda_alloc(c, sizeof(T), loc);
            self.add_ref();
        }

        void init(const array_t &that) {
            // init by refcount
            self._info = that._info;
            self._data = that._data;
            self.add_ref();
        }

        void assign(const array_t &that) {
            // assign by refcount
            that.add_ref();
            self.release();
            self._info = that._info;
            self._data = that._data;
        }

        void copy(const array_t &a) {
            // init _data with the content of a
            // THIS array can be shorter than 'a' NEVER longer
            // size_t n=size();
            // for (int i=0; i<n; ++i) _data[i] = a._data[i];
            if (self._info->c < a._info->n)
                throw std::bad_alloc();

            size_t n = a._info->n;
            self._info->n = n;
            cuda_copy(self._data, self._info->loc, a._data, a._info->loc, n*sizeof(T));
        }

    public:

        // ------------------------------------------------------------------
        // types
        typedef T                   value_type;
        typedef value_type &        reference;
        typedef const value_type &  const_reference;
        typedef value_type*         pointer;
        typedef const value_type*   const_pointer;
        typedef size_t              size_type;
        typedef ptrdiff_t           difference_type;

    public:
        // ------------------------------------------------------------------
        // constructors

        /// Create an empty array, with max_size 0
        array_t(): array_t(0, 0){ }

        /// Create an array with the specified max_size and size.
        /// If 'n' is -1, size and capacity/max_size will be the same
        explicit array_t(size_t n, loc_t loc): array_t(n, n, loc) {}
        array_t(int c, int n, loc_t loc) {
            alloc(c, n, loc);
        }

        /// Create an array by reference
        array_t(const array_t &that): array_t(that, false) {}

        /// Create an array by cloning
        array_t(const array_t &that, bool clone) {
            if (clone) {
                size_t n = that._info->n;
                alloc(n, n, that._info->loc);
                copy(that);
            }
            else {
                init(that);
            }
        }

        ~array_t(){ release(); }

        // ------------------------------------------------------------------
        // operations

        /// Clone the current array
        array_t  clone() const { return array_t(self, true); }
        array_t& to(loc_t to_loc) {
            loc_t curr_loc = self._info->loc;
            size_t bytes = self._info->n * sizeof(T);

            if (curr_loc == loc_t::unified) {
                // it is not necessary to move the memory
            }
            elsif (curr_loc == to_loc) {
                // memory already located in the correct position
            }
            elsif (curr_loc == loc_t::host_mapped && to_loc == loc_t::device) {
                self._info->ptr = self._data;
                void *copy = cuda_copy(nullptr, loc_t::device_mapped, self._data, curr_loc, bytes);
                self._data = reinterpret_cast<T*>(copy);
                self._info->loc = loc_t::device_mapped;
            }
            elsif (curr_loc == loc_t::device_mapped && to_loc == loc_t::host) {
                self._data = reinterpret_cast<T*>(self._info->ptr);
                self._info->loc = loc_t::host_mapped;
            }
            elsif (curr_loc != to_loc) {
                void *copy = cuda_alloc(self._info->n, sizeof(T), to_loc);
                cuda_copy(copy, to_loc, self._data, curr_loc, bytes);
                cuda_free(self._data, curr_loc);
                self._data = reinterpret_cast<T*>(copy);
                self._info->loc = to_loc;
            }
            else {
                // throw cuda_error(CUresult::CUDA_ERROR_ILLEGAL_ADDRESS);
            }
            return self;
        }

        // ------------------------------------------------------------------
        // properties

        /// array size, n of elements (<= capacity)
        [[nodiscard]] size_t size()     const { return self._info->n; }
        /// array max_size/capacity
        [[nodiscard]] size_t max_size() const { return self._info->c; }
        /// if the array is empty (size == 0)
        [[nodiscard]] bool   empty()    const { return self._info->n == 0; }
        /// location of the array
        [[nodiscard]] loc_t  loc()      const { return self._info->loc; }

        /// return the pointer to the internal data
        [[nodiscard]] T*& data() const { return self._data; }

        // ------------------------------------------------------------------
        // accessors
        // at(i) supports negative indices

        T &at(int i)       { return self._data[i>=0 ? i : size()+i]; }
        T  at(int i) const { return self._data[i>=0 ? i : size()+i]; }

        T &operator[](size_t i)       { return self._data[i]; }
        T  operator[](size_t i) const { return self._data[i]; }

        // ------------------------------------------------------------------
        // assignment

        array_t &operator =(const array_t &that) {
            if (this == &that){} // disable warning
            assign(that);
            return self;
        }

    };

    template<typename T>
    void copy(array_t<T>& dst, const array_t<T>& src) {
        dst.copy(src);
    }
}

#endif //CUDA_CUDAMEM_H
