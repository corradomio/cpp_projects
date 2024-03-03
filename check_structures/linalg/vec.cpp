//
// Created by Corrado Mio on 23/02/2024.
//
#include "linalg.h"
#include <cmath>

using namespace stdx::linalg;


// -------------------------------------------------------------------
// data_p
// -------------------------------------------------------------------

void data_p::alloc(size_t n) {
    p = (data_t*)new char[sizeof(data_t) + n * sizeof(float)];
    p->refc = 0;
    p->size = n;
    d = p->data;
    add_ref();
}

void data_p::init(const data_p& that) {
    self.p = that.p;
    self.d = that.d;
    self.add_ref();
}


void data_p::fill(float s) {
    size_t n = size();
    float *d = self.data();
    for(int i=0; i<n; ++i)
        d[i] = s;
}


void data_p::fill(const data_p& that) {
    check(self, that);
    size_t n = size();
    float *s = that.data();
    float *d = self.data();
    for (int i=0; i<n; ++i)
        d[i] = s[i];
}


void data_p::assign(const data_p& that) {
    that.add_ref();
    self.release();
    self.p = that.p;
    self.d = that.d;
}


// --------------------------------------------------------------------------

// void data_p::add_eq(float s) {
//     size_t n = size();
//     float *d = data();
//     for (int i=0; i<n; ++i)
//         d[i] += s;
// }
//
//
// void data_p::sub_eq(float s) {
//     size_t n = size();
//     float *d = data();
//     for (int i=0; i<n; ++i)
//         d[i] -= s;
// }
//
//
// void data_p::mul_eq(float s) {
//     size_t n = size();
//     float *d = data();
//     for (int i=0; i<n; ++i)
//         d[i] *= s;
// }
//
//
// void data_p::div_eq(float s) {
//     size_t n = size();
//     float *d = data();
//     for (int i=0; i<n; ++i)
//         d[i] /= s;
// }
//
//
// void data_p::neg_eq() {
//     size_t n = size();
//     float *d = data();
//     for (int i=0; i<n; ++i)
//         d[i] = -d[i];
// }

// --------------------------------------------------------------------------

// void data_p::add_eq(const data_p& that) {
//     check(self, that);
//     size_t n = size();
//     float *s = that.data();
//     float *d = self.data();
//     for (int i=0; i<n; ++i)
//         d[i] += s[i];
// }
//
//
// void data_p::sub_eq(const data_p& that) {
//     check(self, that);
//     size_t n = size();
//     float *s = that.data();
//     float *d = self.data();
//     for (int i=0; i<n; ++i)
//         d[i] -= s[i];
// }
//
//
// void data_p::mul_eq(const data_p& that) {
//     check(self, that);
//     size_t n = size();
//     float *s = that.data();
//     float *d = self.data();
//     for (int i=0; i<n; ++i)
//         d[i] *= s[i];
// }
//
//
// void data_p::div_eq(const data_p& that) {
//     check(self, that);
//     size_t n = size();
//     float *s = that.data();
//     float *d = self.data();
//     for (int i=0; i<n; ++i)
//         d[i] /= s[i];
// }
//
//
// void data_p::lin_eq(float a, const data_p& that) {
//     check(self, that);
//     size_t n = size();
//     float *s = that.data();
//     float *d = self.data();
//     for (int i=0; i<n; ++i)
//         d[i] += a*s[i];
// }

// --------------------------------------------------------------------------

// v = f(v)
void data_p::apply_eq(float (*fun)(float)) {
    size_t n = size();
    float *d = data();
    for (int i=0; i<n; ++i)
        d[i] += fun(d[i]);
}

// v = f(v, that)
void data_p::apply_eq(float (*fun)(float, float), const data_p& that) {
    check(self, that);
    size_t n = size();
    float *s = that.data();
    float *d = self.data();
    for (int i=0; i<n; ++i)
        d[i] += fun(d[i], s[i]);
}

// v = f(v, s)
void data_p::apply_eq(float (*fun)(float, float), float s) {
    size_t n = size();
    float *d = data();
    for (int i=0; i<n; ++i)
        d[i] += fun(d[i], s);
}

// v = f(v, s)
void data_p::apply_eq(float (*fun)(float, float, float), float f, const data_p& that) {
check(self, that);
    size_t n = size();
    float *s = that.data();
    float *d = data();
    for (int i=0; i<n; ++i)
        d[i] += fun(d[i], f, s[i]);
}


// -------------------------------------------------------------------
// vector
// -------------------------------------------------------------------
// constructor

vector::vector() {
    alloc(0);
}

vector::vector(size_t n) {
    alloc(n);
    fill(0);
}

vector::vector(float s, size_t n) {
    alloc(n);
    fill(s);
};

vector::vector(const vector& that, bool clone) {
    if (clone) {
        alloc(that.size());
        fill(that);
    }
    else {
        init(that);
    }
}

// -------------------------------------------------------------------
// functions

// -------------------------------------------------------------------
// assignments

vector stdx::linalg::operator +(float s, const vector& v) { vector r(s, v.size()); r.apply_eq(add, v); return r; }
vector stdx::linalg::operator -(float s, const vector& v) { vector r(s, v.size()); r.apply_eq(sub, v); return r; }
vector stdx::linalg::operator *(float s, const vector& v) { vector r(s, v.size()); r.apply_eq(mul, v); return r; }
vector stdx::linalg::operator /(float s, const vector& v) { vector r(s, v.size()); r.apply_eq(div, v); return r; }


// -------------------------------------------------------------------
// operations

float vector::dot (const vector& v) const {
    check(self, v);
    size_t n = size();
    float *d = data();
    float *s = v.data();
    float res = 0;
    for (int i=0; i<n; ++i)
        res += d[i]*s[i];
    return res;
}

vector vector::dot(const matrix& m) const {
    check(self, m);
    size_t cols = m.cols();
    size_t rows = m.rows();
    vector r(cols);

    for (int i=0; i<cols; ++i) {
        float s = 0;
        for (int j=0; j<rows; ++j)
            s += at(j)*m.at(j, i);
        r.at(i) = s;
    }

    return r;
}

// -------------------------------------------------------------------
// functions


// -------------------------------------------------------------------

vector stdx::linalg::range(size_t n) {
    vector r(n);
    for (int i=0; i<n; ++i)
        r[i] = float(i);
    return r;
}


const vector& vector::print() const {
    size_t n = size();
    float *s = data();
    if (n == 0)
        printf("[]\n");
    else {
        printf("[%.3f", p->data[0]);
        for (int i=1; i<n; ++i)
            printf(", %.3f", p->data[i]);
        printf("]\n");
    }

    return self;
}




