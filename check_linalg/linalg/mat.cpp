//
// Created by Corrado Mio on 24/02/2024.
//
#include "linalg.h"
#include <cmath>

using namespace stdx::linalg;


// -------------------------------------------------------------------
// constructor

matrix::matrix() {
    alloc(0);
    c = 0;
}

matrix::matrix(size_t n) {
    alloc(n*n);
    c = n;
    init(0);
}

matrix::matrix(size_t n, size_t m) {
    alloc(n*m);
    c = m;
    init(0);
}

matrix::matrix(float s, size_t n, size_t m) {
    alloc(n*m);
    c = m;
    init(s);
}

matrix::matrix(const matrix& m, bool clone) {
    if (clone) {
        alloc(m.size());
        c = m.c;
        init(m);
    }
    else {
        self.p = m.p;
        self.c = m.c;
        add_ref();
    }
}


matrix matrix::reshape(size_t n, size_t m) const {
    size_t sz = size();
    if (n == -1) n = sz/m;
    if (m == -1) m = sz/n;
    if (n*m != sz)
        throw bad_dimensions();

    matrix r(self);
    r.c = m;
    return r;
}

matrix matrix::transpose() const {
    size_t rows = self.rows();
    size_t cols = self.cols();
    matrix r(cols, rows);

    for (int i=0; i<rows; ++i)
        for (int j=0; j<cols; ++j)
            r.at(j,i) = self.at(i,j);
    return r;
}


// -------------------------------------------------------------------
// check

// void matrix::check(const matrix& m) const {
//     if (cols() != m.cols())
//         throw bad_dimensions();
//     if (size() != m.size())
//         throw bad_dimensions();
// }
//
// void matrix::check_m(const matrix& m) const {
//     if (cols() != m.rows())
//         throw bad_dimensions();
// }
//
// void matrix::check_v(const vector& v) const {
//     if (cols() != v.size())
//         throw bad_dimensions();
// }
//
// void matrix::check_u(const vector& v) const {
//     if (v.size() != rows())
//         throw bad_dimensions();
// }

// -------------------------------------------------------------------

// void matrix::add_eq(float s) {
//     size_t n = size();
//     float *d = data();
//     for (int i=0; i<n; ++i)
//         d[i] += s;
// }
//
//
// void matrix::sub_eq(float s) {
//     size_t n = size();
//     float *d = data();
//     for (int i=0; i<n; ++i)
//         d[i] -= s;
// }
//
//
// void matrix::mul_eq(float s) {
//     size_t n = size();
//     float *d = data();
//     for (int i=0; i<n; ++i)
//         d[i] *= s;
// }
//
//
// void matrix::div_eq(float s) {
//     size_t n = size();
//     float *d = data();
//     for (int i=0; i<n; ++i)
//         d[i] /= s;
// }
//
//
// void matrix::neg_eq() {
//     size_t n = size();
//     float *d = data();
//     for (int i=0; i<n; ++i)
//         d[i] = -d[i];
// }
//
//
//
// void matrix::add_eq(const matrix& m) {
//     check(m);
//     size_t n = size();
//     float *s = m.data();
//     float *d =  data();
//     for (int i=0; i<n; ++i)
//         d[i] += s[i];
// }
//
//
// void matrix::sub_eq(const matrix& m) {
//     check(m);
//     size_t n = size();
//     float *s = m.data();
//     float *d =  data();
//     for (int i=0; i<n; ++i)
//         d[i] -= s[i];
// }
//
//
// void matrix::mul_eq(const matrix& m) {
//     check(m);
//     size_t n = size();
//     float *s = m.data();
//     float *d =  data();
//     for (int i=0; i<n; ++i)
//         d[i] *= s[i];
// }
//
//
// void matrix::div_eq(const matrix& m) {
//     check(m);
//     size_t n = size();
//     float *s = m.data();
//     float *d =  data();
//     for (int i=0; i<n; ++i)
//         d[i] /= s[i];
// }
//
//
// void matrix::lin_eq(float a, const matrix& m) {
//     check(m);
//     size_t n = size();
//     float *s = m.data();
//     float *d =  data();
//     for (int i=0; i<n; ++i)
//         d[i] += a*s[i];
// }


// -------------------------------------------------------------------
// assignment

matrix& matrix::operator =(const matrix& m) {
    m.add_ref();
    release();
    self.p = m.p;
    self.c = m.c;
    return self;
}

// -------------------------------------------------------------------
// operations

matrix matrix::dot(const matrix& m) const {
    check_dot(self, m);
    size_t rows = this->rows();
    size_t cross = this->cols();
    size_t cols = m.cols();
    matrix r(rows, cols);

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; ++j) {
            float s = 0;
            for(int k=0; k<cross; ++k)
                s += at(i,j)*m.at(k,j);
            r.at(i,j) = s;
        }
    }
    return r;
}

vector matrix::dot(const vector& v) const {
    check(self, v);
    size_t rows = this->rows();
    size_t cols = this->cols();
    vector r(rows);

    for (int i=0; i<rows; ++i) {
        float s = 0;
        for (int j=0; j<cols; ++j)
            s += at(i,j)*v.at(j);
        r.at(i) = s;
    }
    return r;
}

// -------------------------------------------------------------------

matrix stdx::linalg::identity(size_t n) {
    matrix r(n);
    for(int i=0; i<n; ++i)
        r.at(i,i) = 1;
    return r;
}

matrix stdx::linalg::range(size_t n, size_t m) {
    matrix r(n, m);
    size_t s = n*m;
    for (int i=0; i<s; ++i)
        r.at(i) = float(i);
    return r;
}

// -------------------------------------------------------------------

matrix stdx::linalg::operator +(float s, const matrix& m) { matrix r(s, m.rows(), m.cols()); r.add_eq(m); return r; }
matrix stdx::linalg::operator -(float s, const matrix& m) { matrix r(s, m.rows(), m.cols()); r.sub_eq(m); return r; }
matrix stdx::linalg::operator *(float s, const matrix& m) { matrix r(s, m.rows(), m.cols()); r.mul_eq(m); return r; }
matrix stdx::linalg::operator /(float s, const matrix& m) { matrix r(s, m.rows(), m.cols()); r.div_eq(m); return r; }

// -------------------------------------------------------------------
// debug

const matrix& matrix::print() const {
    size_t rows = self.rows();
    size_t cols = self.cols();
    if (rows == 0) {
        printf("[]\n");
        return self;
    }
    printf("[\n");
    for (int i=0; i<rows; ++i) {
        if (cols == 0)
            printf("  []\n");
        else {
            printf("  [%.3f", at(i, 0));
            for (int j=1; j<cols; ++j)
                printf(" %.3f", at(i, j));
            printf("]\n");
        }
    }
    printf("]\n");
    return self;
}


// -------------------------------------------------------------------
// vector cross product

matrix vector::cross(const vector& v) const {
    size_t rows = size();
    size_t cols = v.size();
    matrix r(rows, cols);

    for(int i=0; i<rows; ++i)
        for(int j=0; j<cols; ++j)
            r.at(i,j) = at(i)*v.at(j);
    return r;
}





