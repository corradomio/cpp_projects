//
// Created by Corrado Mio on 29/02/2024.
//

#ifndef STDX_ARRAY_OP_H
#define STDX_ARRAY_OP_H

#include <exception>
#include <stdexcept>
#include <cmath>
#include "../array.h"

/*
 *  It must ebe possible to write expressions using scalar values OR arrays with EXACTLY the same syntax.
 *
 *  r = [initial value]
 *  r += | -= | *= | /= b
 *  r = a + | - | * | / b
 *  r = a[i]
 *  a[i] = r
 *
 */

namespace stdx {

    template<typename T>    T neg(T x)      { return -x; }
    template<typename T>    T  sq(T x)      { return x*x; }
    template<typename T>    T sum(T x, T y) { return x+y; }
    template<typename T>    T sub(T x, T y) { return x-y; }
    template<typename T>    T mul(T x, T y) { return x*y; }
    template<typename T>    T div(T x, T y) { return x/y; }

    // ----------------------------------------------------------------------

    template<typename T>
    void check(const stdx::array_t<T>& a, const stdx::array_t<T>& b) {
        if (a.size() != b.size())
            throw std::range_error("Different sizes");
    }

    // ----------------------------------------------------------------------

    template<typename T>
    stdx::array_t<T> zeros(size_t n) {
        stdx::array_t<T> a(n);
        a = 0;
        return a;
    }

    template<typename T>
    stdx::array_t<T> ones(size_t n) {
        stdx::array_t<T> a(n);
        a = 1;
        return a;
    }

    // ----------------------------------------------------------------------

    /// a = fun(a)
    /// example:   a = -a
    template<typename T>
    stdx::array_t<T>& apply_eq(T (*fun)(T), stdx::array_t<T>& a) {
        size_t n = a.size();
        T *d = a.data();
        for(int i=0; i<n; ++i)
            d[i] = fun(d[i]);
        return a;
    }

    /// a = fun(a, s)
    /// example:    a = pow(a, e)
    template<typename T>
    stdx::array_t<T>& apply_eq(T (*fun)(T, T), stdx::array_t<T>& a, T s) {
        size_t n = a.size();
        T *d = a.data();
        for(int i=0; i<n; ++i)
            d[i] = fun(d[i], s);
        return a;
    }

    /// a = fun(a, b)
    /// example:    a = a+b
    template<typename T>
    stdx::array_t<T>& apply_eq(T (*fun)(T, T), stdx::array_t<T>& a, const stdx::array_t<T>& b) {
        check(a, b);
        size_t n = a.size();
        T *d = a.data();
        T *s = b.data();
        for(int i=0; i<n; ++i)
            d[i] = fun(d[i], s[i]);
        return a;
    }

    /// a = fun(a, c, b)
    /// example:    a = a + c*b
    template<typename T>
    stdx::array_t<T>& apply_eq(T (*fun)(T, T, T), stdx::array_t<T>& a, T c, const stdx::array_t<T>& b) {
        check(a, b);
        size_t n = a.size();
        T *d = a.data();
        T *s = b.data();
        for(int i=0; i<n; ++i)
            d[i] = fun(d[i], c, s[i]);
        return a;
    }

    // ----------------------------------------------------------------------

    template<typename T>
    stdx::array_t<T>& operator +=(stdx::array_t<T>& a, const stdx::array_t<T>& b) {
        return apply_eq(sum, a, b);
    }

    template<typename T>
    stdx::array_t<T>& operator -=(stdx::array_t<T>& a, const stdx::array_t<T>& b) {
        return apply_eq(sub, a, b);
    }

    template<typename T>
    stdx::array_t<T>& operator *=(stdx::array_t<T>& a, const stdx::array_t<T>& b) {
        return apply_eq(mul, a, b);
    }

    template<typename T>
    stdx::array_t<T>& operator /=(stdx::array_t<T>& a, const stdx::array_t<T>& b) {
        return apply_eq(div, a, b);
    }

    // ----------------------------------------------------------------------

    template<typename T>
    stdx::array_t<T>& operator +=(stdx::array_t<T>& a, T b) {
        return apply_eq(sum, a, b);
    }

    template<typename T>
    stdx::array_t<T>& operator -=(stdx::array_t<T>& a, T b) {
        return apply_eq(sub, a, b);
    }

    template<typename T>
    stdx::array_t<T>& operator *=(stdx::array_t<T>& a, T b) {
        return apply_eq(mul, a, b);
    }

    template<typename T>
    stdx::array_t<T>& operator /=(stdx::array_t<T>& a, T b) {
        return apply_eq(div, a, b);
    }

    // ----------------------------------------------------------------------

    template<typename T>
    stdx::array_t<T> operator -(const stdx::array_t<T>& a) {
        stdx::array_t<T> r = a.clone();
        return apply_eq(r, neg);
    }


    template<typename T>
    stdx::array_t<T> operator +(const stdx::array_t<T>& a, const stdx::array_t<T>& b) {
        stdx::array_t<T> r = a.clone();
        return apply_eq(sum, r, b);
    }

    template<typename T>
    stdx::array_t<T> operator -(const stdx::array_t<T>& a, const stdx::array_t<T>& b) {
        stdx::array_t<T> r = a.clone();
        return apply_eq(sub, r, b);
    }

    template<typename T>
    stdx::array_t<T> operator *(const stdx::array_t<T>& a, const stdx::array_t<T>& b) {
        stdx::array_t<T> r = a.clone();
        return apply_eq(mul, r, b);
    }

    template<typename T>
    stdx::array_t<T> operator /(const stdx::array_t<T>& a, const stdx::array_t<T>& b) {
        stdx::array_t<T> r = a.clone();
        return apply_eq(div, r, b);
    }

    // ----------------------------------------------------------------------

    template<typename T>
    stdx::array_t<T> operator +(const stdx::array_t<T>& a, T b) {
        stdx::array_t<T> r = a.clone();
        return apply_eq(sum, r, b);
    }

    template<typename T>
    stdx::array_t<T> operator -(const stdx::array_t<T>& a, T b) {
        stdx::array_t<T> r = a.clone();
        return apply_eq(sub, r, b);
    }

    template<typename T>
    stdx::array_t<T> operator *(const stdx::array_t<T>& a, T b) {
        stdx::array_t<T> r = a.clone();
        return apply_eq(mul, r, b);
    }

    template<typename T>
    stdx::array_t<T> operator /(const stdx::array_t<T>& a, T b) {
        stdx::array_t<T> r = a.clone();
        return apply_eq(div, r, b);
    }

}

#endif //STDX_ARRAY_OP_H
