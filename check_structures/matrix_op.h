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

#include "matrix.h"

namespace stdx {

    template<typename T>
    void check(const matrix_t<T> &a, const matrix_t<T> &b) {
        if (a.rows() != b.cols() or a.cols() != b.cols())
            throw std::range_error("Incompatible dimensions");
    }

    template<typename T>
    void check(const matrix_t<T> &a, const vector_t<T> &b) {
        if (a.cols() != b.size())
            throw std::range_error("Incompatible dimensions");
    }

    template<typename T>
    void check(const vector_t<T> &a, const matrix_t<T> &b) {
        if (a.size() != b.rows())
            throw std::range_error("Incompatible dimensions");
    }

    template<typename T>
    void check_dot(const matrix_t<T> &a, const matrix_t<T> &b) {
        if (a.cols() != b.rows())
            throw std::range_error("Incompatible dimensions");
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
        m = T(1);
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

    template<typename V, typename T>
    matrix_t<T> operator+(V a, const matrix_t<T>  &b) {
        matrix_t<T> r(b.rows(), b.cols());
        r = a;
        apply_eq(sum, r, b);
        return r;
    }

    template<typename V, typename T>
    matrix_t<T> operator-(V a, const matrix_t<T>  &b) {
        matrix_t<T> r(b.rows(), b.cols());
        r = a;
        apply_eq(sub, r, b);
        return r;
    }

    template<typename V, typename T>
    matrix_t<T> operator*(V a, const matrix_t<T>  &b) {
        matrix_t<T> r(b.rows(), b.cols());
        r = a;
        apply_eq(mul, r, b);
        return r;
    }

    template<typename V, typename T>
    matrix_t<T> operator/(V a, const matrix_t<T>  &b) {
        matrix_t<T> r(b.rows(), b.cols());
        r = a;
        apply_eq(div, r, b);
        return r;
    }

    // ----------------------------------------------------------------------


}

#endif //STDX_MATRIX_OP_H
