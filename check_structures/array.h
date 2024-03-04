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

#include <cstddef>

#ifndef self
#define self (*this)
# endif

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

    namespace detail {

        struct info_t {
            size_t refc;    // refcount
            size_t n;       // size
            size_t c;       // capacity

            info_t(size_t c, size_t n) : refc(0), c(c), n(n) {}
        };
    }

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

        detail::info_t *_info;
        T *_data;

        void add_ref() const { _info->refc++; }
        void release() { if (0 == --_info->refc) {
            delete   _info;
            delete[] _data;
        }}

        void alloc(size_t c, size_t n) {
            n = (n == -1) ? c : n;

            // c must be multiple than MIN_ALLOC
            size_t d = c % MIN_ALLOC;
            c = c + (d ? (MIN_ALLOC - d) : 0);

            _info = new detail::info_t(c, n);
            _data = new T[c];
            add_ref();
        }

        void init(const array_t &that) {
            // assign by refcount
            _info = that._info;
            _data = that._data;
            add_ref();
        }

        void assign(const array_t &that) {
            // assign by refcount
            that.add_ref();
            release();
            _info = that._info;
            _data = that._data;
        }

        void insert(const T &s, size_t i) {
            size_t n = _info->n;
            if (i < 0) i = n+i;
            for (size_t j=n; j>i; --j)
                _data[j] = _data[j-1];
            _data[i] = s;
            _info->n++;
        }

        void fill(const T &s) {
            // init _data with s
            size_t n=size();
            for (int i=0; i<n; ++i) _data[i] = s;
        }

        void fill(const array_t &a) {
            // init _data with the content of a
            // THIS array can be shorter than 'a' NEVER longer
            size_t n=size();
            for (int i=0; i<n; ++i) _data[i] = a._data[i];
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
        /// If 'n' is -1, size and max_size will be the same
        explicit array_t(size_t n): array_t(n, n) {}
        array_t(int c, int n) {
            alloc(c, n);
            fill(T());
        }

        /// Create an array by reference
        array_t(const array_t &that): array_t(that, false) {}

        /// Create an arry by cloning
        array_t(const array_t &that, bool clone) {
            if (clone) {
                alloc(that.size(), that.size());
                fill(that);
            }
            else {
                init(that);
            }
        }

        ~array_t(){ release(); }

        /// Clone the current array
        array_t clone() const { return array_t(self, true); }

        /// Ensure that the array has a single reference.
        /// If it has multiple references, it is cloned
        array_t norefs() { return (_info->refc == 1) ? self : self.clone(); }

        // ------------------------------------------------------------------
        // properties

        /// array size, n of elements (<= capacity)
        [[nodiscard]] size_t size()     const { return _info->n; }
        /// array capacity
        [[nodiscard]] size_t max_size() const { return _info->c; }
        /// if the array is empty (size == 0)
        [[nodiscard]] bool   empty()    const { return _info->n == 0; }

        [[nodiscard]] T* data() const { return _data; }

        // ------------------------------------------------------------------
        // accessors
        // at(i) supports negative indices

        T &at(int i)       { return _data[i>=0 ? i : size()+i]; }
        T  at(int i) const { return _data[i>=0 ? i : size()+i]; }

        T &operator[](size_t i)       { return _data[i]; }
        T  operator[](size_t i) const { return _data[i]; }

        // ------------------------------------------------------------------
        // assignment

        array_t &operator =(const T &s) {
            fill(s);
            return self;
        }

        array_t &operator =(const array_t &a) {
            if (this == &a){} // disable warning
            assign(a);
            return self;
        }

        // ------------------------------------------------------------------
        // operations

        /// change size, AND capacity if size > current capacity
        void size(size_t n) {
            if (n > _info->c) max_size(n);
            _info->n = n;
        }

        /// change capacity ONLY if new capacity > current capacity
        void max_size(size_t c) {
            if (c <= max_size()) return;

            size_t n = size();  // keep current size
            array_t t(self);    // to avoid de-allocation on release()
            release();          // 't' contains an extra reference of the current array
            alloc(c, n);        // allocate the new capacity (same size)
            fill(t);            // copy the content of t
        }

    };

    // check compatibility
    template<typename T>
    void check(const  array_t<T>& a, const  array_t<T>& b) {
        if (a.size() != b.size())
            throw std::range_error("Incompatible dimensions");
    }

    // apply_eq functions

    template<typename T>
     void apply_eq(T (*fun)(T),  array_t<T>& a) {
        T* d = a.data();
        size_t n = a.size();

        for (int i=0; i<n; ++i)
            d[i] = fun(d[i]);
    }

    template<typename T>
     void apply_eq(T (*fun)(T, T),  array_t<T>& a, T s) {
        T* y = a.data();
        size_t n = a.size();
        for (int i=0; i<n; ++i)
            y[i] = fun(y[i], s);
    }

    template<typename T>
     void apply_eq(T (*fun)(T, T),  array_t<T>& a, const  array_t<T>& b) {
        check(a, b);

        T* y = a.data();
        T* x = b.data();
        size_t n = a.size();
        for (int i=0; i<n; ++i)
            y[i] = fun(y[i], x[i]);
    }

    template<typename T>
     void apply_eq(T (*fun)(T, T, T),  array_t<T>& a, T s, const  array_t<T>& b) {
        check(a, b);

        T* y = a.data();
        T* x = b.data();
        size_t n = a.size();
        for (int i=0; i<n; ++i)
            y[i] = fun(y[i], s, x[i]);
    }

    // apply functions

    template<typename T>
    array_t<T> apply(T (*fun)(T),  array_t<T>& a) {
        array_t<T> r = a.clone();
        apply_eq(fun, r);
        return r;
    }

    template<typename T>
    array_t<T> apply(T (*fun)(T, T),  array_t<T>& a, T s) {
        array_t<T> r = a.clone();
        apply_eq(fun, r, s);
        return r;
    }

    template<typename T>
    array_t<T> apply(T (*fun)(T, T),  array_t<T>& a, const  array_t<T>& b) {
        array_t<T> r = a.clone();
        apply_eq(fun, r, b);
        return r;
    }

    template<typename T>
    array_t<T> apply(T (*fun)(T, T, T),  array_t<T>& a, T s, const  array_t<T>& b) {
        array_t<T> r = a.clone();
        apply_eq(fun, r, s, b);
        return r;
    }
}

#endif // STDX_ARRAY_H
