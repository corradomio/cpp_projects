//
// Created by Corrado Mio on 29/02/2024.
//
// C++ casts
//      static_cast
//      dynamic_cast
//      const_cast
//      reinterpret_cast


#ifndef STDX_MATRIX_OP_H
#define STDX_MATRIX_OP_H

#include "../array.h"
#include "matrix.h"

namespace stdx::linalg {

    // ----------------------------------------------------------------------
    // check

    template<typename T>
    void check(const matrix_t<T> &m1, const matrix_t<T> &m2) {
        if (m1.rows() != m2.cols() or m1.cols() != m2.cols())
            throw bad_dimensions();
    }

    // ----------------------------------------------------------------------

    template<typename T>
    matrix_t<T> zeros(size_t rows, size_t cols) {
        matrix_t<T> m(rows, cols);
        m = 0;
        return m;
    }

    template<typename T>
    matrix_t<T> ones(size_t rows, size_t cols) {
        matrix_t<T> m(rows, cols);
        m = 1;
        return m;
    }

    template<typename T>
    matrix_t<T> identity(size_t rows, size_t cols=-1) {
        if (cols == -1) cols = rows;
        size_t n = std::min(rows, cols);
        matrix_t<T> m(rows, cols);
        for (int i=0; i<n; ++i)
            m [i,i]= 1;
        return m;
    }

    // ----------------------------------------------------------------------
    // m += m
    // m -= m
    // m += m
    // m /= m

    template<typename T>
    matrix_t<T> &operator+=(matrix_t<T> &a, const matrix_t<T> &b) {
        apply_eq(sum, a, b);
        return a;
    }

    template<typename T>
    matrix_t<T> &operator-=(matrix_t<T> &a, const matrix_t<T> &b) {
        apply_eq(sub, a, b);
        return a;
    }

    template<typename T>
    matrix_t<T> &operator*=(matrix_t<T> &a, const matrix_t<T> &b) {
        apply_eq(mul, a, b);
        return a;
    }

    template<typename T>
    matrix_t<T> &operator/=(matrix_t<T> &a, const matrix_t<T> &b) {
        apply_eq(div, a, b);
        return a;
    }

    // ----------------------------------------------------------------------
    // m += v
    // m -= v
    // m += v
    // m /= v

    template<typename T>
    matrix_t<T> &operator+=(matrix_t<T> &a, T b) {
        apply_eq(sum, a, T(b));
        return a;
    }

    template<typename T>
    matrix_t<T> &operator-=(matrix_t<T> &a, T b) {
        apply_eq(sub, a, T(b));
        return a;
    }

    template<typename T>
    matrix_t<T> &operator*=(matrix_t<T> &a, T b) {
        apply_eq(mul, a, T(b));
        return a;
    }

    template<typename T>
    matrix_t<T> &operator/=(matrix_t<T> &a, T b) {
        apply_eq(div, a, T(b));
        return a;
    }

    // ----------------------------------------------------------------------
    // -m
    // m+m
    // m-m
    // m*m
    //m /m

    template<typename T>
    matrix_t<T> operator-(const matrix_t<T> &a) {
        matrix_t<T> r = a.clone();
        apply_eq(r, neg);
        return r;
    }

    template<typename T>
    matrix_t<T> operator+(const matrix_t<T> &a, const matrix_t<T> &b) {
        matrix_t<T> r = a.clone();
        apply_eq<T>(sum, r, b);
        return r;
    }

    template<typename T>
    matrix_t<T> operator-(const matrix_t<T> &a, const matrix_t<T> &b) {
        matrix_t<T> r = a.clone();
        apply_eq(sub, r, b);
        return r;
    }

    template<typename T>
    matrix_t<T> operator*(const matrix_t<T> &a, const matrix_t<T> &b) {
        matrix_t<T> r = a.clone();
        apply_eq(mul, r, b);
        return r;
    }

    template<typename T>
    matrix_t<T> operator/(const matrix_t<T> &a, const matrix_t<T> &b) {
        matrix_t<T> r = a.clone();
        apply_eq(div, r, b);
        return r;
    }

    // ----------------------------------------------------------------------
    // m + v
    // m - v
    // m * v
    // m / v

    template<typename T>
    matrix_t<T> operator+(const matrix_t<T> &a, T b) {
        matrix_t<T> r = a.clone();
        apply_eq(sum, r, T(b));
        return r;
    }

    template<typename T>
    matrix_t<T> operator-(const matrix_t<T> &a, T b) {
        matrix_t<T> r = a.clone();
        apply_eq(sub, r, T(b));
        return r;
    }

    template<typename T>
    matrix_t<T> operator*(const matrix_t<T> &a, T b) {
        matrix_t<T> r = a.clone();
        apply_eq(mul, r, T(b));
        return r;
    }

    template<typename T>
    matrix_t<T> operator/(const matrix_t<T> &a, T b) {
        matrix_t<T> r = a.clone();
        apply_eq(div, r, T(b));
        return r;
    }

    // ----------------------------------------------------------------------
    // v + m
    // v - m
    // v * m
    // v / m

    template<typename T>
    matrix_t<T> operator+(T a, const matrix_t<T>  &b) {
        matrix_t<T> r(b.rows(), b.cols());
        r = a;
        apply_eq(sum, r, b);
        return r;
    }

    template<typename T>
    matrix_t<T> operator-(T a, const matrix_t<T>  &b) {
        matrix_t<T> r(b.rows(), b.cols());
        r = a;
        apply_eq(sub, r, b);
        return r;
    }

    template<typename T>
    matrix_t<T> operator*(T a, const matrix_t<T>  &b) {
        matrix_t<T> r(b.rows(), b.cols());
        r = a;
        apply_eq(mul, r, b);
        return r;
    }

    template<typename T>
    matrix_t<T> operator/(T a, const matrix_t<T>  &b) {
        matrix_t<T> r(b.rows(), b.cols());
        r = a;
        apply_eq(div, r, b);
        return r;
    }

    // ----------------------------------------------------------------------

}

#endif //STDX_MATRIX_OP_H
