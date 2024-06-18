//
// Created by Corrado Mio on 03/06/2024.
//

#ifndef CUDA_CUDAMEM_H
#define CUDA_CUDAMEM_H

#include <cstdlib>
#include <stdexcept>
#include <cuda.h>
#include <language.h>
#include "cudacpp.h"

#define MIN_ALLOC 16

namespace cudacpp {

    /// Where the allocated memory is locates
    ///
    ///     - host: host memory
    ///     - host_locked: host memory but the memory is 'locked' to speedup
    ///         memory transfers
    ///     - host_mapped: host memory mapped into the device address space
    ///         but using a different address
    ///     - device mapped: as host_mapped but from th point of view of the
    ///         device. It is not available for the allocation
    ///     - unified: device memory mapped into the host address space
    ///         using the same address
    ///
    enum loc_t {
        host, device, host_locked,
        host_mapped, device_mapped,
        unified
    };

    struct info_t {
        size_t refc;    // refcount
        size_t n;       // size
        size_t c;       // capacity
        loc_t  loc;     // location
        void*  ptr;     // saved host pointer for host_mapped when converted into
                        // device_mapped
        void*  data;    // copy of pointer the allocated memory

        info_t(size_t n, loc_t l) : refc(0), c(n), n(n), loc(l) {}
        info_t(size_t c, size_t n, loc_t l) : refc(0), c(c), n(n), loc(l) {}
    };

    /// Alloc a block of memory to contain n elements of esize bytes each one
    /// located in the specified location (host/device)
    ///
    /// \param n num of elements
    /// \param esize element size in bytes
    /// \param loc where to locate the memory
    /// \return the allocated block of memory
    void* cuda_alloc(size_t n, size_t esize, loc_t loc);

    /// Free the block of memory located located in the specified location (host/device)
    ///
    /// \param p memory pointer (in host or device)
    /// \param loc where the memory is located
    /// \return nullptr
    void* cuda_free(void* p, loc_t loc);
    void* cuda_free(info_t* info);

    /// Copy the content of src (located at src_loc) into the memory identified by dst
    /// and located at dst_loc
    ///
    /// \param dst pointer to destination memory
    /// \param to_loc destination memory location
    /// \param src pointer to source memory
    /// \param src_loc source memory location
    /// \param size num of bytes to copy
    /// \return
    void* cuda_copy(void* dst, loc_t to_loc, void* src, loc_t src_loc, size_t size);

    /// move the content of info to the location to_loc. The element size is
    /// necessary because data->n is specified in number of elements
    void* cuda_move(loc_t to_loc, info_t* info, size_t esize);

    /// fill the allocated memory dst, located at dst_loc, with the value src,
    /// of esize btes
    ///
    /// \param dst pointer to destination memory
    /// \param dst_loc destination memory location
    /// \param dst_size num of bytes to the destination
    /// \param src source value (max 4 bytes)
    /// \param esize num of bytes of the source value
    void cuda_fill(void* dst, loc_t dst_loc, size_t dst_size, int src, size_t esize);

    /// Reference counted vector transferable between host & device memory
    ///
    template<typename T>
    struct array_t {
        // DOESN'T change the order!
        T *_data;           // it MUST BE the same than '_info->data'
        info_t *_info;

        void add_ref() const { self._info->refc++; }
        void release() { if (0 == --self._info->refc) {
            // if (self._info->loc == device_mapped) {
            //     self._data = reinterpret_cast<T*>(self._info->ptr);
            //     self._info->loc = host_mapped;
            // }
            // cuda_free(self._data, self._info->loc);
            cuda_free(self._info);
            delete   self._info;
        }}

        void alloc(size_t c, size_t n, loc_t loc) {
            n = (n == -1) ? c : n;

            // c must be multiple than MIN_ALLOC
            size_t d = c % MIN_ALLOC;
            c = c + (d ? (MIN_ALLOC - d) : 0);

            self._info = new info_t(c, n, loc);
            self._info->data = cuda_alloc(c, sizeof(T), loc);
            self._data = (T*) self._info->data;
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
            // THIS array can be longer than 'a' NEVER shorter
            if (self._info->c < a._info->n)
                throw std::bad_alloc();

            size_t n = a._info->n;
            self._info->n = n;
            self._data = (T*)cuda_copy(self._info, self._info->loc, a._data, a._info->loc,
                                       n * sizeof(T));
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
        array_t(size_t c, size_t n, loc_t loc) {
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
            self._data  = (T*) cuda_move(to_loc, self._info, sizeof(T));
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
        [[nodiscard]] T*     data()     const { return self._data; }

        // ------------------------------------------------------------------
        // accessors
        // at(i) supports negative indices

        T &at(int64_t i)       { return self._data[i>=0 ? i : size()+i]; }
        T  at(int64_t i) const { return self._data[i>=0 ? i : size()+i]; }

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
