//
// Created by Corrado Mio on 29/02/2024.
//

#ifndef STDX_VECTOR_H
#define STDX_VECTOR_H

#include "../array.h"

namespace stdx::linalg {

    template<typename T> struct vector_t;
    template<typename T> struct matrix_t;


    template<typename T>
    struct vector_t : public stdx::array_t<T> {

        void assign(const vector_t &that) {
            array_t<T>::assign(that);
        }

        // ------------------------------------------------------------------
        // constructor

        vector_t() { }
        explicit vector_t(size_t n): array_t<T>(n, n) { }

        // implicit conversion
        vector_t(const array_t<T> &v): vector_t(v, false){ }

        // implicit conversion
        vector_t(const array_t<T> &v, bool clone): array_t<T>(v, clone){ }


        // ------------------------------------------------------------------
        // operations

        vector_t clone() const { return vector_t(self, true); }

        vector_t norefs() const { return  (self._info->refc == 1) ? self : self.clone(); }

        // ------------------------------------------------------------------
        // accessors

        // T &at(int i)       { return self._data[i>=0 ? i : self.size()+i]; }
        // T  at(int i) const { return self._data[i>=0 ? i : self.size()+i]; }

        T &operator[](size_t i)       { return self._data[i]; }
        T  operator[](size_t i) const { return self._data[i]; }

        // ------------------------------------------------------------------
        // assignment

        vector_t &operator =(const array_t<T> &v) {
            assign(v);
            return self;
        }

        vector_t &operator =(T s) {
            self.fill(s);
            return self;
        }

        // ------------------------------------------------------------------
        // dot

        T dot(const vector_t& v) const;
        vector_t<T> dot(const matrix_t<T>& m) const;

    };

}

#endif //STDX_VECTOR_H
