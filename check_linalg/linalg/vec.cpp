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
    add_ref();
}


void data_p::init(float s) {
    size_t n = size();
    float *d = data();
    for(int i=0; i<n; ++i)
        d[i] = s;
}


void data_p::init(const data_p& o) {
    check(self, o);
    size_t n = size();
    float *s = o.data();
    float *d =   data();
    for (int i=0; i<n; ++i)
        d[i] = s[i];
}


void data_p::assign(const data_p& o) {
    o.add_ref();
    self.release();
    self.p = o.p;
}


// --------------------------------------------------------------------------

void data_p::add_eq(float s) {
    size_t n = size();
    float *d =   data();
    for (int i=0; i<n; ++i)
        d[i] += s;
}


void data_p::sub_eq(float s) {
    size_t n = size();
    float *d =   data();
    for (int i=0; i<n; ++i)
        d[i] -= s;
}


void data_p::mul_eq(float s) {
    size_t n = size();
    float *d =   data();
    for (int i=0; i<n; ++i)
        d[i] *= s;
}


void data_p::div_eq(float s) {
    size_t n = size();
    float *d =   data();
    for (int i=0; i<n; ++i)
        d[i] /= s;
}


void data_p::neg_eq() {
    size_t n = size();
    float *d = data();
    for (int i=0; i<n; ++i)
        d[i] = -d[i];
}

// --------------------------------------------------------------------------

void data_p::add_eq(const data_p& o) {
    check(self, o);
    size_t n = size();
    float *s = o.data();
    float *d =   data();
    for (int i=0; i<n; ++i)
        d[i] += s[i];
}


void data_p::sub_eq(const data_p& o) {
    check(self, o);
    size_t n = size();
    float *s = o.data();
    float *d =   data();
    for (int i=0; i<n; ++i)
        d[i] -= s[i];
}


void data_p::mul_eq(const data_p& o) {
    check(self, o);
    size_t n = size();
    float *s = o.data();
    float *d =   data();
    for (int i=0; i<n; ++i)
        d[i] *= s[i];
}


void data_p::div_eq(const data_p& o) {
    check(self, o);
    size_t n = size();
    float *s = o.data();
    float *d =   data();
    for (int i=0; i<n; ++i)
        d[i] /= s[i];
}


void data_p::lin_eq(float a, const data_p& o) {
    check(self, o);
    size_t n = size();
    float *d = data();
    float *s = o.data();
    for (int i=0; i<n; ++i)
        d[i] += a*s[i];
}


void data_p::apply_eq(float (*f)(float)) {
    size_t n = size();
    float *d = data();
    for (int i=0; i<n; ++i)
        d[i] += f(d[i]);
}

void data_p::apply_eq(float (*f)(float, float), const data_p& o) {
    check(self, o);
    size_t n = size();
    float *d =   data();
    float *s = o.data();
    for (int i=0; i<n; ++i)
        d[i] += f(d[i], s[i]);
}

void data_p::apply_eq(float (*f)(float, float), float b) {
    size_t n = size();
    float *d = data();
    for (int i=0; i<n; ++i)
        d[i] += f(d[i], b);
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
    init(0);
}

vector::vector(float s, size_t n) {
    alloc(n);
    init(s);
};

vector::vector(const vector& v, bool clone) {
    if (clone) {
        alloc(v.size());
        init(v);
    }
    else {
        self.p = v.p;
        add_ref();
    }
}

// -------------------------------------------------------------------
// functions

float sqf(float x) { return x*x; }

vector stdx::linalg::abs(const vector& v) { vector r=v.clone(); r.apply_eq(::fabsf); return r; }
vector stdx::linalg::log(const vector& v) { vector r=v.clone(); r.apply_eq(::logf); return r; }
vector stdx::linalg::exp(const vector& v) { vector r=v.clone(); r.apply_eq(::expf); return r; }
vector stdx::linalg::pow(const vector& v, float e) { vector r=v.clone(); r.apply_eq(::powf, e); return r; }
vector stdx::linalg::sq( const vector& v)  { vector r=v.clone(); r.apply_eq(::sqf); return r; }
vector stdx::linalg::sqrt(const vector& v) { vector r=v.clone(); r.apply_eq(::sqrtf); return r; }

vector stdx::linalg:: sin(const vector& v) { vector r=v.clone(); r.apply_eq(::sinf); return r; }
vector stdx::linalg:: cos(const vector& v) { vector r=v.clone(); r.apply_eq(::cosf); return r; }
vector stdx::linalg:: tan(const vector& v) { vector r=v.clone(); r.apply_eq(::tanf); return r; }
vector stdx::linalg::asin(const vector& v) { vector r=v.clone(); r.apply_eq(::asinf); return r; }
vector stdx::linalg::acos(const vector& v) { vector r=v.clone(); r.apply_eq(::acosf); return r; }
vector stdx::linalg::atan(const vector& v) { vector r=v.clone(); r.apply_eq(::atanf); return r; }
vector stdx::linalg::atan(const vector& u, const vector& v) {
    vector r=u.clone(); r.apply_eq(::atan2f, v);
    return r;
}

// -------------------------------------------------------------------
// assignments


vector stdx::linalg::operator +(float s, const vector& v) { vector r(s, v.size()); r.add_eq(v); return r; }
vector stdx::linalg::operator -(float s, const vector& v) { vector r(s, v.size()); r.sub_eq(v); return r; }
vector stdx::linalg::operator *(float s, const vector& v) { vector r(s, v.size()); r.mul_eq(v); return r; }
vector stdx::linalg::operator /(float s, const vector& v) { vector r(s, v.size()); r.div_eq(v); return r; }


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

// vector stdx::linalg::abs(const vector& v) {
//     vector r(v);
//     size_t n = v.size();
//     float *s =  v.data();
//     float *d = r.data();
//     for (int i=0; i<n; ++i)
//         d[i] = ::fabsf(s[i]);
//     return r;
// }
//
// vector stdx::linalg::log(const vector& v) {
//     vector r(v);
//     size_t n = v.size();
//     float *s =  v.data();
//     float *d = r.data();
//     for (int i=0; i<n; ++i)
//         d[i] = ::logf(s[i]);
//     return r;
// }
//
// vector stdx::linalg::exp(const vector& v) {
//     vector r(v);
//     size_t n = v.size();
//     float *s =  v.data();
//     float *d = r.data();
//     for (int i=0; i<n; ++i)
//         d[i] = ::expf(s[i]);
//     return r;
// }
//
// vector stdx::linalg::power(const vector& v, float e) {
//     vector r(v);
//     size_t n = v.size();
//     float *s =  v.data();
//     float *d = r.data();
//     for (int i=0; i<n; ++i)
//         d[i] = ::powf(s[i], e);
//     return r;
// }
//
// vector stdx::linalg::sq(const vector& v) {
//     vector r(v);
//     size_t n = v.size();
//     float *s =  v.data();
//     float *d = r.data();
//     for (int i=0; i<n; ++i)
//         d[i] = s[i]*s[i];
//     return r;
// }
//
// vector stdx::linalg::sqrt(const vector& v) {
//     vector r(v);
//     size_t n = v.size();
//     float *s =  v.data();
//     float *d = r.data();
//     for (int i=0; i<n; ++i)
//         d[i] = ::sqrtf(s[i]);
//     return r;
// }
//
// vector  stdx::linalg::sin(const vector& v) {
//     vector r(v);
//     size_t n = v.size();
//     float *s =  v.data();
//     float *d = r.data();
//     for (int i=0; i<n; ++i)
//         d[i] = ::sinf(s[i]);
//     return r;
// }
//
// vector  stdx::linalg::cos(const vector& v) {
//     vector r(v);
//     size_t n = v.size();
//     float *s =  v.data();
//     float *d = r.data();
//     for (int i=0; i<n; ++i)
//         d[i] = ::cosf(s[i]);
//     return r;
// }
//
// vector  stdx::linalg::tan(const vector& v) {
//     vector r(v);
//     size_t n = v.size();
//     float *s =  v.data();
//     float *d = r.data();
//     for (int i=0; i<n; ++i)
//         d[i] = ::tanf(s[i]);
//     return r;
// }
//
// vector stdx::linalg::asin(const vector& v) {
//     vector r(v);
//     size_t n = v.size();
//     float *s =  v.data();
//     float *d = r.data();
//     for (int i=0; i<n; ++i)
//         d[i] = ::asinf(s[i]);
//     return r;
// }
//
// vector stdx::linalg::acos(const vector& v) {
//     vector r(v);
//     size_t n = v.size();
//     float *s =  v.data();
//     float *d = r.data();
//     for (int i=0; i<n; ++i)
//         d[i] = ::acosf(s[i]);
//     return r;
// }
//
// vector stdx::linalg::atan(const vector& v) {
//     vector r(v);
//     size_t n = v.size();
//     float *s =  v.data();
//     float *d = r.data();
//     for (int i=0; i<n; ++i)
//         d[i] = ::atanf(s[i]);
//     return r;
// }
//
// vector stdx::linalg::atan(const vector& u, const vector& v) {
//     vector r(v);
//     size_t n = v.size();
//     float *s = u.data();
//     float *t = v.data();
//     float *d = r.data();
//     for (int i=0; i<n; ++i)
//         d[i] = ::atan2f(s[i], t[i]);
//     return r;
// }



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




