//
// Created by Corrado Mio on 29/02/2024.
//

#ifndef STDX_VECTOR_OP_H
#define STDX_VECTOR_OP_H

#include <exception>
#include <stdexcept>
#include "vector.h"

namespace stdx {

    template<typename T>
    T neg(T x) { return -x; }

    template<typename T>
    T sq(T x) { return x * x; }

    template<typename T>
    T sum(T x, T y) { return x + y; }

    template<typename T>
    T sub(T x, T y) { return x - y; }

    template<typename T>
    T mul(T x, T y) { return x * y; }

    template<typename T>
    T div(T x, T y) { return x / y; }

    // ----------------------------------------------------------------------

    template<typename T>
    vector_t<T> zeros(size_t n) {
        vector_t<T> a(n);
        a = 0;
        return a;
    }

    template<typename T>
    vector_t<T> ones(size_t n) {
        vector_t<T> a(n);
        a = 1;
        return a;
    }

    // ----------------------------------------------------------------------
    // v += v
    // v -= v
    // v *= v
    // v /= v

    template<typename T>
    vector_t<T> &operator+=(vector_t<T> &a, const vector_t<T> &b) {
        apply_eq(sum, a, b);
        return a;
    }

    template<typename T>
    vector_t<T> &operator-=(vector_t<T> &a, const vector_t<T> &b) {
        apply_eq(sub, a, b);
        return a;
    }

    template<typename T>
    vector_t<T> &operator*=(vector_t<T> &a, const vector_t<T> &b) {
        apply_eq(mul, a, b);
        return a;
    }

    template<typename T>
    vector_t<T> &operator/=(vector_t<T> &a, const vector_t<T> &b) {
        apply_eq(div, a, b);
        return a;
    }

    // ----------------------------------------------------------------------
    // v += s
    // v -= s
    // v *= s
    // v /= s

    template<typename T>
    vector_t<T> &operator+=(vector_t<T> &a, T b) {
        apply_eq(sum, a, T(b));
        return a;
    }

    template<typename T>
    vector_t<T> &operator-=(vector_t<T> &a, T b) {
        apply_eq(sub, a, T(b));
        return a;
    }

    template<typename T>
    vector_t<T> &operator*=(vector_t<T> &a, T b) {
        apply_eq(mul, a, T(b));
        return a;
    }

    template<typename T>
    vector_t<T> &operator/=(vector_t<T> &a, T b) {
        apply_eq(div, a, T(b));
        return a;
    }

    // ----------------------------------------------------------------------
    // v + v
    // v - v
    // v * v
    // v / v

    template<typename T>
    vector_t<T> operator-(const vector_t<T> &a) {
        vector_t<T> r = a.clone();
        return apply_eq(r, neg);
    }


    template<typename T>
    vector_t<T> operator+(const vector_t<T> &a, const vector_t<T> &b) {
        vector_t<T> r = a.clone();
        return apply_eq(sum, r, b);
    }

    template<typename T>
    vector_t<T> operator-(const vector_t<T> &a, const vector_t<T> &b) {
        vector_t<T> r = a.clone();
        return apply_eq(sub, r, b);
    }

    template<typename T>
    vector_t<T> operator*(const vector_t<T> &a, const vector_t<T> &b) {
        vector_t<T> r = a.clone();
        return apply_eq(mul, r, b);
    }

    template<typename T>
    vector_t<T> operator/(const vector_t<T> &a, const vector_t<T> &b) {
        vector_t<T> r = a.clone();
        return apply_eq(div, r, b);
    }

    // ----------------------------------------------------------------------
    // v + s
    // v - s
    // v * s
    // v / s

    template<typename T>
    vector_t<T> operator+(const vector_t<T> &a, T b) {
        vector_t<T> r = a.clone();
        return apply_eq(sum, r, T(b));
    }

    template<typename T>
    vector_t<T> operator-(const vector_t<T> &a, T b) {
        vector_t<T> r = a.clone();
        return apply_eq(sub, r, T(b));
    }

    template<typename T>
    vector_t<T> operator*(const vector_t<T> &a, T b) {
        vector_t<T> r = a.clone();
        return apply_eq(mul, r, T(b));
    }

    template<typename T>
    vector_t<T> operator/(const vector_t<T> &a, T b) {
        vector_t<T> r = a.clone();
        return apply_eq(div, r, T(b));
    }

    // ----------------------------------------------------------------------
    // s + v
    // s - v
    // s * v
    // s / v

    template<typename V, typename T>
    vector_t<T> operator+(V a, const vector_t<T> & b) {
        vector_t<T> r(b.size());
        r = a;
        return apply_eq(sum, r, b);
    }

    template<typename V, typename T>
    vector_t<T> operator-(V a, const vector_t<T> & b) {
        vector_t<T> r(b.size());
        r = a;
        return apply_eq(sub, r, b);
    }

    template<typename V, typename T>
    vector_t<T> operator*(V a, const vector_t<T> & b) {
        vector_t<T> r(b.size());
        r = a;
        return apply_eq(mul, r, b);
    }

    template<typename V, typename T>
    vector_t<T> operator/(V a, const vector_t<T> & b) {
        vector_t<T> r(b.size());
        r = a;
        return apply_eq(div, r, b);
    }

    // ----------------------------------------------------------------------
    // dot(u,v)

    template<typename T>
    T dot(const vector_t<T> & a, const vector_t<T> & b) {
        check(a, b);
        T s = 0;
        size_t n = a.size();
        T* x = a.data();
        T* y = b.data();
        for (int i=0; i<n; ++i)
            s += x[i]*y[i];
        return s;
    }

}

#endif //STDX_VECTOR_OP_H
