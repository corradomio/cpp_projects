//
// Created by Corrado Mio on 14/02/2024.
//

#ifndef CHECK_LINALG_LINALG_H
#define CHECK_LINALG_LINALG_H

#include <cstddef>
#include <exception>
#include <stdexcept>

#define self (*this)

namespace stdx::linalg {

    // ----------------------------------------------------------------------
    // exceptions
    // ----------------------------------------------------------------------

    struct bad_dimensions : public std::runtime_error {
        bad_dimensions() : runtime_error("") { }
    };

    // ----------------------------------------------------------------------
    // data_t
    // ----------------------------------------------------------------------

    struct data_t {
        mutable size_t refc;
        size_t  size;
        float   data[0];
    };

    // ----------------------------------------------------------------------
    // data_p
    // ----------------------------------------------------------------------

    struct data_p {
        data_t *p;

        // -------------------------------------------------------------------
        // refcount

        void add_ref() const { p->refc++; }
        void release() const { if (0 == --p->refc) delete p; }

        // -------------------------------------------------------------------
        // operations

        void alloc(size_t n);
        void init(float s);
        void init(const data_p& o);
        void check(const data_p& o) const;

        // -------------------------------------------------------------------
        // properties

        [[nodiscard]] inline size_t size() const { return p->size; }
        [[nodiscard]] inline float* data() const { return p->data; }

        // -------------------------------------------------------------------
        // operations

        [[nodiscard]] inline float& at(size_t i)       { return p->data[i]; }
        [[nodiscard]] inline float  at(size_t i) const { return p->data[i]; }

    };

    // ----------------------------------------------------------------------
    // vector
    // ----------------------------------------------------------------------

    struct vector;
    struct matrix;

    // ----------------------------------------------------------------------
    // vector
    // ----------------------------------------------------------------------
    // r = c
    // r = u+v u-v u*v u/v      element-wise
    // r = s*u u*s u/s          scalar
    // s = u.v                  (dot) scalar
    // r = v.M                  (dot) vector / matrix
    // r = u + s*v              linear

    struct vector: public data_p {

        void check(const vector& v) const;
        void check(const matrix& m) const;

        // -------------------------------------------------------------------
        // constructor & destructor

        vector();
        explicit vector(size_t n);
        vector(float s, size_t n);
        vector(const vector& v);
        ~vector() { release(); }

        [[nodiscard]] vector clone() const;

        // -------------------------------------------------------------------
        // access

        float& operator[](size_t i)       { return at(i); }
        float  operator[](size_t i) const { return at(i); }

        // -------------------------------------------------------------------
        // operations

        void add_eq(float s);
        void sub_eq(float s);
        void mul_eq(float s);
        void div_eq(float s);
        void neg_eq();

        void add_eq(const vector& o);
        void sub_eq(const vector& o);
        void mul_eq(const vector& o);
        void div_eq(const vector& o);
        void lin_eq(float a, const vector& o);

        // -------------------------------------------------------------------
        // assignment

        vector& operator  = (float s) { init(s);   return self; }
        vector& operator += (float s) { add_eq(s); return self; }
        vector& operator -= (float s) { sub_eq(s); return self; }
        vector& operator *= (float s) { mul_eq(s); return self; }
        vector& operator /= (float s) { div_eq(s); return self; }

        vector& operator  = (const vector& v);
        vector& operator += (const vector& v) { add_eq(v); return self; }
        vector& operator -= (const vector& v) { sub_eq(v); return self; }
        vector& operator *= (const vector& v) { mul_eq(v); return self; }
        vector& operator /= (const vector& v) { div_eq(v); return self; }

        vector& linear_eq (float a, const vector& v) { lin_eq(a, v); return self; }

        // -------------------------------------------------------------------
        // operations

        vector operator +() const { vector r(self);             return r; }
        vector operator -() const { vector r(self); r.neg_eq(); return r; }

        vector  operator +(const vector& v) const { vector r(self); r.add_eq(v); return r; }
        vector  operator -(const vector& v) const { vector r(self); r.sub_eq(v); return r; }
        vector  operator *(const vector& v) const { vector r(self); r.mul_eq(v); return r; }
        vector  operator /(const vector& v) const { vector r(self); r.div_eq(v); return r; }

        [[nodiscard]] vector linear(float a, const vector& v) const { vector r(self); r.lin_eq(a, v); return self; }

        [[nodiscard]] float  dot (const vector& v) const;
        [[nodiscard]] vector dot(const matrix& m) const;
        [[nodiscard]] matrix cross(const vector& v) const;


        void print();
    };

    inline vector zeros(size_t n) { return vector{0, n}; }
    inline vector  ones(size_t n) { return vector{1, n}; }
    vector range(size_t n);

    // ----------------------------------------------------------------------
    // operations

    vector operator +(float s, const vector& v);// { vector r(s, v.size()); r.add_eq(v); return r; }
    vector operator -(float s, const vector& v);// { vector r(s, v.size()); r.sub_eq(v); return r; }
    vector operator *(float s, const vector& v);// { vector r(s, v.size()); r.mul_eq(v); return r; }
    vector operator /(float s, const vector& v);// { vector r(s, v.size()); r.div_eq(v); return r; }

    // ----------------------------------------------------------------------
    // functions

    vector abs(const vector& v);
    vector log(const vector& v);    // base e
    vector exp(const vector& v);    // e^v
    vector power(const vector& v, float e); // v^e
    vector sq(const vector& v);     // v^2
    vector sqrt(const vector& v);   // square root

    vector  sin(const vector& v);
    vector  cos(const vector& v);
    vector  tan(const vector& v);
    vector asin(const vector& v);
    vector acos(const vector& v);
    vector atan(const vector& v);
    vector atan(const vector& u, const vector& v);

    // ----------------------------------------------------------------------
    // matrix
    // ----------------------------------------------------------------------
    // R = c
    // R = A+B A-B A*B A/B  element-wise
    // R = A.B
    // r = A.v
    // r = v.A
    // R = A + s*B

    struct matrix : public data_p {
        size_t c; // n of columns

        void check  (const matrix& m) const;
        void check_m(const matrix& m) const;
        void check_v(const vector& v) const;
        void check_u(const vector& v) const;

        // ----------------------------------------------------------------------
        // constructor

        matrix();
        explicit matrix(size_t n);
        matrix(size_t n, size_t m);
        matrix(float s, size_t n, size_t m);
        matrix(const matrix& m);

        [[nodiscard]] matrix clone() const;
        [[nodiscard]] matrix reshape(size_t n, size_t m) const;
        [[nodiscard]] matrix transpose() const;

        // ----------------------------------------------------------------------
        // properties

        [[nodiscard]] size_t rows() const { return size()/c; }
        [[nodiscard]] size_t cols() const { return c; }

        // -------------------------------------------------------------------
        // access

        [[nodiscard]] inline float& at(size_t i)                 { return p->data[i]; }
        [[nodiscard]] inline float  at(size_t i)           const { return p->data[i]; }
        [[nodiscard]] inline float& at(size_t i, size_t j)       { return p->data[i*c + j]; }
        [[nodiscard]] inline float  at(size_t i, size_t j) const { return p->data[i*c + j]; }

        [[nodiscard]] inline float& operator[](size_t i, size_t j)       { return at(i, j); }
        [[nodiscard]] inline float  operator[](size_t i, size_t j) const { return at(i, j); }

        // -------------------------------------------------------------------
        // operations

        void add_eq(float s);
        void sub_eq(float s);
        void mul_eq(float s);
        void div_eq(float s);
        void neg_eq();

        void add_eq(const matrix& m);
        void sub_eq(const matrix& m);
        void mul_eq(const matrix& m);
        void div_eq(const matrix& m);
        void lin_eq(float a, const matrix& m);

        // ----------------------------------------------------------------------
        // assignment

        matrix& operator  = (float s) { init(s);   return self; }
        matrix& operator += (float s) { add_eq(s); return self; }
        matrix& operator -= (float s) { sub_eq(s); return self; }
        matrix& operator *= (float s) { mul_eq(s); return self; }
        matrix& operator /= (float s) { div_eq(s); return self; }

        matrix& operator  = (const matrix& m);
        matrix& operator += (const matrix& m) { add_eq(m); return self;}
        matrix& operator -= (const matrix& m) { sub_eq(m); return self;}
        matrix& operator *= (const matrix& m) { mul_eq(m); return self;}
        matrix& operator /= (const matrix& m) { div_eq(m); return self;}

        matrix& linear_eq (float a, const matrix& m) { lin_eq(a, m); return self;}

        // -------------------------------------------------------------------
        // operations

        matrix operator +()                 const { matrix r(self);              return r;}
        matrix operator -()                 const { matrix r(self); r.neg_eq( ); return r; }

        matrix  operator +(const matrix& m) const { matrix r(self); r.add_eq(m); return r; }
        matrix  operator -(const matrix& m) const { matrix r(self); r.sub_eq(m); return r; }
        matrix  operator *(const matrix& m) const { matrix r(self); r.mul_eq(m); return r; }
        matrix  operator /(const matrix& m) const { matrix r(self); r.div_eq(m); return r; }

        [[nodiscard]] matrix linear(float a, const matrix& m) const {
            matrix r(self);
            r.lin_eq(a, m);
            return r;
        }

        [[nodiscard]] matrix dot(const matrix& m) const;
        [[nodiscard]] vector dot(const vector& v) const;

        void print();
    };

    matrix identity(size_t n);
    inline matrix zeros(size_t n, size_t m) { return matrix{0, n, m}; }
    inline matrix  ones(size_t n, size_t m) { return matrix{1, n, m}; }
    matrix range(size_t n, size_t m);

    // ----------------------------------------------------------------------
    // operations

    matrix operator +(float s, const matrix& m);// { matrix r(s); r.add_eq(m); return r; }
    matrix operator -(float s, const matrix& m);// { matrix r(s); r.sub_eq(m); return r; }
    matrix operator *(float s, const matrix& m);// { matrix r(s); r.mul_eq(m); return r; }
    matrix operator /(float s, const matrix& m);// { matrix r(s); r.div_eq(m); return r; }

}

#endif //CHECK_LINALG_LINALG_H
