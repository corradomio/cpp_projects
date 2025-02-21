//
// Created by Corrado Mio on 25/02/2024.
//
/*
std::array           template < class T, size_t N > class array;

 Member types

    value_type          T
    reference	        value_type&
    const_reference	    const value_type&
    pointer	            value_type*
    const_pointer	    const value_type*
    size_type           size_t
    difference_type	p   trdiff_t

    iterator
    const_iterator
    reverse_iterator        reverse_iterator<iterator>
    const_reverse_iterator  reverse_iterator<const_iterator>

 Member iterators

     begin(),  end(),  cbegin(),  cend(),
    rbegin(), rend(), crbegin(), crend()

 Capacity

    size()
    max_size()
    empty()

 Accessors
    operator[](i)
    at(i)
    front()
    back()
    _data()

 Modifiers

    fill(T v)
    swap(array &that)
 */

#ifndef STDX_ARRAY_H
#define STDX_ARRAY_H

#include <stdexcept>
#include <cstddef>
#include "language.h"
#include "exceptions.h"

#define MIN_ALLOC 16

namespace stdx {

    // ----------------------------------------------------------------------
    // operations

    template<typename T> T neg(T x)      { return -x;    }
    template<typename T> T sq(T x)       { return x * x; }
    template<typename T> T sum(T x, T y) { return x + y; }
    template<typename T> T sub(T x, T y) { return x - y; }
    template<typename T> T mul(T x, T y) { return x * y; }
    template<typename T> T div(T x, T y) { return x / y; }

    // end
    // ----------------------------------------------------------------------

}

namespace stdx {

    ///
    /// The implementation uses two objects (_info  &_data) to simplify the implementation
    /// of arrays with types having a not empty constructor/destructor. In this way
    /// it will be responsibility of the compiler to generate the code to all them.
    /// The alternative was to allocate a single block of memory, and to use the placement
    /// new and delete to call them manually.
    ///
    /// \tparam T
    template<typename T>
    struct array_t {
    private:

        struct info_t {
            size_t refc;    // refcount
            size_t n;       // size
            size_t c;       // capacity

            info_t(size_t c, size_t n) : refc(0), c(c), n(n) {}
        };

        // DOESN'T change the order!
        T *_data;
        array_t::info_t *_info;

        void add_ref() const { self._info->refc++; }
        void release() { if (0 == --self._info->refc) {
            delete[] self._data;
            delete   self._info;
        }}

        void alloc(size_t c, size_t n) {
            n = (n == -1) ? c : n;

            // c must be multiple than MIN_ALLOC
            size_t d = c % MIN_ALLOC;
            c = c + (d ? (MIN_ALLOC - d) : 0);

            self._info = new array_t::info_t(c, n);
            self._data = new T[c];
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

        // void insert(const T &s, size_t i) {
        //     size_t n = _info->n;
        //     if (i < 0) i = n+i;
        //     for (size_t j=n; j>i; --j)
        //         _data[j] = _data[j-1];
        //     _data[i] = s;
        //     _info->n++;
        // }

        void fill(const T &s) {
            // init _data with s
            size_t n=size();
            for (size_t i=0; i<n; ++i) self._data[i] = s;
        }

        void copy(const array_t &a) {
            // init _data with the content of a
            // THIS array can be shorter than 'a' NEVER longer
            // size_t n=size();
            // for (size_t i=0; i<n; ++i) _data[i] = a._data[i];
            size_t n = self._info->n;
            memcpy(self._data, a._data, n*sizeof(T));
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
        explicit array_t(size_t n): array_t(n, n) {}
        array_t(size_t c, size_t n) {
            alloc(c, n);
        }

        /// Create an array by cloning
        array_t(const array_t &that, bool clone=false) {
            if (clone) {
                size_t n = that._info->n;
                alloc(n, n);
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
        array_t clone() const { return array_t(self, true); }

        // /// Ensure that the array has a single reference.
        // /// If it has multiple references, it is cloned
        // array_t norefs() const { return (self._info->refc == 1) ? self : self.clone(); }

        // ------------------------------------------------------------------
        // properties

        /// array size, n of elements (<= capacity)
        [[nodiscard]] size_t size()     const { return self._info->n; }
        /// array max_size/capacity
        [[nodiscard]] size_t max_size() const { return self._info->c; }
        /// if the array is empty (size == 0)
        [[nodiscard]] bool   empty()    const { return self._info->n == 0; }
        /// return the pointer to the internal data
        [[nodiscard]] T*     data()     const { return self._data; }

        /// return a standalone pointer to the internal data.
        /// the data is clones if this object has a refc > 1
        /// otherwise it is removed from this object
        [[maybe_unused]] T* detach() {
            T* data;

            // 0) it is possible to detach the T[] from THIS object
            //    ONLY if it has a single reference
            if (self._info->refc == 1) {
                // 2) retrieve the data pointer
                data = self._data;
                // 3) replace the data pointer with an empty array
                //    to maintain the data structure 'consistent'
                self._data = new T[0];
                // 4) update the internal data structure consistency
                //    In theory it is not necessary, but it is better
                //    to be safe
                self._info->n = 0;
                self._info->c = 0;
            }
            else {
                // 1) ensure an object with a single reference
                array_t cloned{self, true};
                // 2) retrieve the data pointer
                data = cloned._data;
                // 3) replace the data pointer with an empty array
                //    to maintain the data structure 'consistent'
                cloned._data = new T[0];
                // 4) update the internal data structure consistency
                //    In theory it is not necessary, but it is better
                //    to be safe
                cloned._info->n = 0;
                cloned._info->c = 0;
            }
            // 5) return the data
            return data;
        }

        // ------------------------------------------------------------------
        // accessors
        // at(i) supports negative indices

        T         &at(size_t i)       { return self._data[i]; }
        T          at(size_t i) const { return self._data[i]; }

        T &operator[](size_t i)       { return self._data[i]; }
        T  operator[](size_t i) const { return self._data[i]; }

        // ------------------------------------------------------------------
        // assignment

        array_t &operator =(const T &s) {
            fill(s);
            return self;
        }

        array_t &operator =(const array_t &that) {
            if (this == &that){} // disable warning
            assign(that);
            return self;
        }

        // ------------------------------------------------------------------
        // operations

        /// change size, AND capacity if size > current max_size/capacity
        void size(size_t n) {
            if (n > self._info->c) max_size(n);
            self._info->n = n;
        }

        /// change capacity ONLY if new max_size/capacity > current max_size/capacity
        void max_size(size_t c) {
            if (c <= self._info->c) return;

            size_t n = self._info->n;// keep current size
            array_t t(self);    // to avoid de-allocation on release()
            release();          // 't' contains an extra reference of the current array
            alloc(c, n);        // allocate the new capacity (same size)
            copy(t);         // copy the content of t
        }

        // end
        // ------------------------------------------------------------------

    };

    // check compatibility
    template<typename T>
    void check(const  array_t<T>& a, const  array_t<T>& b) {
        if (a.size() != b.size())
            throw std::range_error("Incompatible dimensions");
    }

    // apply_eq functions
    // a[i] = f(a[i])
    template<typename T>
    void apply_eq(T (*fun)(T),  array_t<T>& a) {
        T* y = a.data();
        size_t n = a.size();

        for (size_t i=0; i<n; ++i)
            y[i] = fun(y[i]);
    }

    // a[i] = f(a[i], s)
    template<typename T>
    void apply_eq(T (*fun)(T, T),  array_t<T>& a, T s) {
        T* y = a.data();
        size_t n = a.size();
        for (size_t i=0; i<n; ++i)
            y[i] = fun(y[i], s);
    }

    // a[i] = f(a[i], b[i])
    template<typename T>
    void apply_eq(T (*fun)(T, T),  array_t<T>& a, const  array_t<T>& b) {
        check(a, b);

        T* y = a.data();
        T* x = b.data();
        size_t n = a.size();
        for (size_t i=0; i<n; ++i)
            y[i] = fun(y[i], x[i]);
    }

    // a[i] = f(a[i], s, b[i])
    template<typename T>
    void apply_eq(T (*fun)(T, T, T),  array_t<T>& a, T s, const  array_t<T>& b) {
        check(a, b);

        T* y = a.data();
        T* x = b.data();
        size_t n = a.size();
        for (size_t i=0; i<n; ++i)
            y[i] = fun(y[i], s, x[i]);
    }

    // apply functions
    // r[i] = f(a[i])
    template<typename T>
    array_t<T> apply(T (*fun)(T),  array_t<T>& a) {
        array_t<T> r(a, true);
        apply_eq(fun, r);
        return r;
    }

    // r[i] = f(a[i], s)
    template<typename T>
    array_t<T> apply(T (*fun)(T, T),  array_t<T>& a, T s) {
        array_t<T> r(a, true);
        apply_eq(fun, r, s);
        return r;
    }

    // r[i] = f(a[i], b[i])
    template<typename T>
    array_t<T> apply(T (*fun)(T, T),  array_t<T>& a, const  array_t<T>& b) {
        array_t<T> r(a, true);
        apply_eq(fun, r, b);
        return r;
    }

    // r[i] = f(a[i], s, b[i])
    template<typename T>
    array_t<T> apply(T (*fun)(T, T, T),  array_t<T>& a, T s, const  array_t<T>& b) {
        array_t<T> r(a, true);
        apply_eq(fun, r, s, b);
        return r;
    }
}

#endif // STDX_ARRAY_H
