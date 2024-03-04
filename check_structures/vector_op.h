//
// Created by Corrado Mio on 29/02/2024.
//

#ifndef STDX_VECTOR_OP_H
#define STDX_VECTOR_OP_H

#include <exception>
#include <stdexcept>
#include "vector.h"

namespace stdx {

    // ----------------------------------------------------------------------
    // check

    template<typename T>
    void check(const vector_t<T> &a, const vector_t<T> &b) {
        if (a.size() != b.size())
            throw std::range_error("Incompatible dimensions");
    }

    // ----------------------------------------------------------------------
    // zeros
    // ones

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
    // v += u
    // v -= u
    // v *= u
    // v /= u

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
    // v + u
    // v - u
    // v * u
    // v / u

    template<typename T>
    vector_t<T> operator-(const vector_t<T> &a) {
        vector_t<T> r = a.clone();
        apply_eq(r, neg);
        return r;
    }


    template<typename T>
    vector_t<T> operator+(const vector_t<T> &a, const vector_t<T> &b) {
        vector_t<T> r = a.clone();
        apply_eq(sum, r, b);
        return r;
    }

    template<typename T>
    vector_t<T> operator-(const vector_t<T> &a, const vector_t<T> &b) {
        vector_t<T> r = a.clone();
        apply_eq(sub, r, b);
        return r;
    }

    template<typename T>
    vector_t<T> operator*(const vector_t<T> &a, const vector_t<T> &b) {
        vector_t<T> r = a.clone();
        apply_eq(mul, r, b);
        return r;
    }

    template<typename T>
    vector_t<T> operator/(const vector_t<T> &a, const vector_t<T> &b) {
        vector_t<T> r = a.clone();
        apply_eq(div, r, b);
        return r;
    }

    // ----------------------------------------------------------------------
    // v + s
    // v - s
    // v * s
    // v / s

    template<typename T>
    vector_t<T> operator+(const vector_t<T> &a, T b) {
        vector_t<T> r = a.clone();
        apply_eq(sum, r, T(b));
        return r;
    }

    template<typename T>
    vector_t<T> operator-(const vector_t<T> &a, T b) {
        vector_t<T> r = a.clone();
        apply_eq(sub, r, T(b));
        return r;
    }

    template<typename T>
    vector_t<T> operator*(const vector_t<T> &a, T b) {
        vector_t<T> r = a.clone();
        apply_eq(mul, r, T(b));
        return r;
    }

    template<typename T>
    vector_t<T> operator/(const vector_t<T> &a, T b) {
        vector_t<T> r = a.clone();
        apply_eq(div, r, T(b));
        return r;
    }

    // ----------------------------------------------------------------------
    // s + v
    // s - v
    // s * v
    // s / v

    template<typename T>
    vector_t<T> operator+(T a, const vector_t<T> & b) {
        vector_t<T> r(b.size());
        r = a;
        apply_eq(sum, r, b);
        return r;
    }

    template<typename T>
    vector_t<T> operator-(T a, const vector_t<T> & b) {
        vector_t<T> r(b.size());
        r = a;
        apply_eq(sub, r, b);
        return r;
    }

    template<typename T>
    vector_t<T> operator*(T a, const vector_t<T> & b) {
        vector_t<T> r(b.size());
        r = a;
        apply_eq(mul, r, b);
        return r;
    }

    template<typename T>
    vector_t<T> operator/(T a, const vector_t<T> & b) {
        vector_t<T> r(b.size());
        r = a;
        apply_eq(div, r, b);
        return r;
    }

    // ----------------------------------------------------------------------
    // dot(u,v)

}

#endif //STDX_VECTOR_OP_H
