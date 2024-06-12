//
// Created by Corrado Mio on 09/06/2024.
//

#ifndef STDX_LINALG_H
#define STDX_LINALG_H

#include <cstdlib>
#include <stdexcept>
#include "language.h"

namespace stdx::linalg {

    template<typename T>
    struct base_t {

    protected:

        struct info_t {
            size_t refc;    // refcount
            size_t n;       // size | n_rows*n_cols
            size_t cols;    // n_cols

            info_t(size_t n) : refc(0), n(n), cols(1) { }
        };

        // DOESN'T change the order!
        T *_data;
        info_t *_info;

        void add_ref() const { self._info->refc++; }

        void release() { if (0 == --self._info->refc) {
            delete[] self._data;
            delete   self._info;
        }}

    protected:

        void alloc(size_t n) {
            // init by allocation
            self._info = new info_t(n);
            self._data = new T[n];
            self.add_ref();
        }

        void init(const base_t &that) {
            // init by refcount
            self._info = that._info;
            self._data = that._data;
            self.add_ref();
        }

        void assign(const base_t &that) {
            // assign by refcount
            that.add_ref();
            self.release();
            self._info = that._info;
            self._data = that._data;
        }

        void fill(const T &s) {
            // init _data with s
            size_t n = self._info->n;
            for (size_t i = 0; i < n; ++i) self._data[i] = s;
        }

        void copy(const base_t &a) {
            // init _data with the content of a
            // the arrays must have the same size
            if (self._info->n != a._info->n)
                throw std::runtime_error("Array size mismatch");
            size_t n = self._info->n;
            memcpy(self._data, a._data, n * sizeof(T));
        }

    public:

        // ------------------------------------------------------------------
        // types
        typedef T value_type;
        typedef value_type &reference;
        typedef const value_type &const_reference;
        typedef value_type *pointer;
        typedef const value_type *const_pointer;
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;

    public:
        // ------------------------------------------------------------------
        // constructors

        explicit base_t(size_t n) { alloc(n); }
        base_t(const base_t &that, bool clone=false) {
            if (clone) {
                size_t n = that._info->n;
                alloc(n);
                copy(that);
            }
            else {
                init(that);
            }
        }

        ~base_t() { release(); }

        // -------------------------------------------------------------------
        // properties

        [[nodiscard]] size_t size() const { return self._info->n; }
        [[nodiscard]] bool  empty() const { return self._info->n == 0; }
        [[nodiscard]] T*    data()  const { return self._data; }


        // -------------------------------------------------------------------
        // special operations

        /// return a standalone pointer to the internal data.
        /// the data is clones if this object has a refc > 1
        /// otherwise it is removed from this object
        [[maybe_unused]] T *detach() {
            T *data;

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
            } else {
                // 1) ensure an object with a single reference
                base_t cloned{self, true};
                // 2) retrieve the data pointer
                data = cloned._data;
                // 3) replace the data pointer with an empty array
                //    to maintain the data structure 'consistent'
                cloned._data = new T[0];
                // 4) update the internal data structure consistency
                //    In theory it is not necessary, but it is better
                //    to be safe
                cloned._info->n = 0;
            }
            // 5) return the data
            return data;
        }

    };

    ///
    /// The implementation uses two objects (_info  &_data) to simplify the implementation
    /// of arrays with types having a not empty constructor/destructor. In this way
    /// it will be responsibility of the compiler to generate the code to all them.
    /// The alternative was to allocate a single block of memory, and to use the placement
    /// new and delete to call them manually.
    ///
    /// \tparam T
    template<typename T>
    struct vector_t: public base_t<T> {

        typedef base_t<T> super;

    public:
        // ------------------------------------------------------------------
        // constructors

        /// Create an empty array, with size 0
        vector_t() : vector_t(0) {}

        /// Create an array with the specified max_size and size.
        explicit vector_t(size_t n): base_t<T>(n) {}

        /// Create an array by cloning
        vector_t(const vector_t &that, bool clone=false): base_t<T>(that, clone) { }

        ~vector_t() {}

        // ------------------------------------------------------------------
        // operations

        /// Clone the current array
        vector_t clone() const { return vector_t(self, true); }

        // /// Ensure that the array has a single reference.
        // /// If it has multiple references, it is cloned
        // vector_t norefs() const { return (self._info->refc == 1) ? self : self.clone(); }

        // ------------------------------------------------------------------
        // properties

        // inherited from base_t

        // ------------------------------------------------------------------
        // accessors

        T         &at(size_t i)       { return self._data[i]; }
        T          at(size_t i) const { return self._data[i]; }

        T &operator[](size_t i)       { return self._data[i]; }
        T  operator[](size_t i) const { return self._data[i]; }

        // ------------------------------------------------------------------
        // assignment

        vector_t &operator=(const T &s) {
            super::fill(s);
            return self;
        }

        vector_t &operator=(const vector_t &that) {
            if (this == &that) {} // disable warning
            super::assign(that);
            return self;
        }

        // end
        // ------------------------------------------------------------------

    };

    // ----------------------------------------------------------------------

    template<typename T>
    struct matrix_t: public base_t<T> {

        typedef base_t<T> super;

    public:
        // ------------------------------------------------------------------
        // constructors

        /// Create an empty matrix, with rows,cols 0,0
        matrix_t() : matrix_t(0, 0) {}

        /// Create a squared matrix
        explicit matrix_t(size_t n): matrix_t(n, n) { }

        /// Create matrix with specified rows and cols.
        matrix_t(size_t n, size_t m): base_t<T>(n*m) {
            self._info->cols = m;
        }

        /// Create a matrix by cloning
        matrix_t(const matrix_t &that, bool clone=false): base_t<T>(that, clone) {
            self._info->cols = that._info->cols;
        }

        ~matrix_t() {}

        // ------------------------------------------------------------------
        // operations

        /// Clone the current matrix
        matrix_t clone() const { return matrix_t(self, true); }

        // /// Ensure that the array has a single reference.
        // /// If it has multiple references, it is cloned
        // matrix_t norefs() const { return (self._info->refc == 1) ? self : self.clone(); }

        // ------------------------------------------------------------------
        // properties

        /// matrix rows and columns
        [[nodiscard]] size_t rows() const { size_t m = self._info->cols; return m ? self._info->n/m : 0; }
        [[nodiscard]] size_t cols() const { return self._info->cols; }

        // ------------------------------------------------------------------
        // accessors

        T         &at(size_t i, size_t j)       { return self._data[i*self._info->cols + j]; }
        T          at(size_t i, size_t j) const { return self._data[i*self._info->cols + j]; }

#if __STDC_VERSION__ >= 202302L
        T &operator[](size_t i, size_t j)       { return self.m[i,j]; }
        T  operator[](size_t i, size_t j) const { return self.m[i,j]; }
#endif
        // ------------------------------------------------------------------
        // assignment

        matrix_t &operator=(const T &s) {
            super::fill(s);
            return self;
        }

        matrix_t &operator=(const matrix_t &that) {
            if (this == &that) {} // disable warning
            super::assign(that);
            return self;
        }

        // end
        // ------------------------------------------------------------------

    };

    // ----------------------------------------------------------------------

    template<typename T>
    struct trmat_t {
        matrix_t<T>& m;
        trmat_t(matrix_t<T>& m): m(m) { }

        // ------------------------------------------------------------------
        // properties

        /// array size, n of elements (<= capacity)
        [[nodiscard]] size_t size() const { return self.m.size(); }
        [[nodiscard]] bool  empty() const { return self.m.empty(); }
        [[nodiscard]] T*    data() const { return self.m.data(); }

        [[nodiscard]] size_t rows() const { return self.m.cols(); }
        [[nodiscard]] size_t cols() const { return self.m.rows(); }
        [[nodiscard]] matrix_t<T> mat() const { return self.m; }

        // ------------------------------------------------------------------
        // accessors
        // at(i) supports negative indices

        T         &at(size_t i, size_t j)       { return self.m.at(j,i); }
        T          at(size_t i, size_t j) const { return self.m.at(j,i); }

#if __STDC_VERSION__ >= 202302L
        T &operator[](size_t i, size_t j)       { return self.m[j,i]; }
        T  operator[](size_t i, size_t j) const { return self.m[j,i]; }
#endif
    };

    // ----------------------------------------------------------------------

    template<typename T>
    matrix_t<T> trmat(const matrix_t<T>& m) {
        size_t rows = m.rows();
        size_t cols = m.cols();
        matrix_t<T>t(cols, rows);
        for (size_t i=0; i<rows; ++i)
            for (size_t j=0; j<cols; ++i)
                t[j, i] = m[i,j];
        return t;
    }

    template<typename T>
    trmat_t<T> tr(matrix_t<T>& m) {
        return trmat_t(m);
    }

    template<typename T>
    matrix_t<T> tr(trmat_t<T>& mt) {
        return mt.mat();
    }
}

#endif //STDX_LINALG_H
