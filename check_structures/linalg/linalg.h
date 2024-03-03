//
// Created by Corrado Mio on 14/02/2024.
//
//  Mathematica
//
//      a + b   Plus        Add Sum
//      a - b   Minus?      Difference Sub
//      a * b   Times       Mul
//      a / b   Quotient?   Div

#ifndef CHECK_LINALG_LINALG_H
#define CHECK_LINALG_LINALG_H

#include <cstddef>
#include <cmath>
#include <exception>
#include <stdexcept>

#ifndef self
#define self (*this)
#endif

namespace stdx::linalg {

    struct data_p;
    struct vector;
    struct matrix;

    // ----------------------------------------------------------------------
    // exceptions
    // ----------------------------------------------------------------------

    struct bad_dimensions : public std::runtime_error {
        bad_dimensions() : runtime_error("") { }
    };

    float neg(float x);
    float add(float x, float y);
    float sub(float x, float y);
    float mul(float x, float y);
    float div(float x, float y);
    float lin(float x, float s, float y);

    void check(const data_p& u, const data_p& v);       // u.size() == v.size()
    void check(const vector& u, const vector& v);       // m.size() == v.size()
    void check(const matrix& m, const vector& v);       // m.cols() == v.size()
    void check(const vector& u, const matrix& m);       // u.size() == m.rows()
    void check(const matrix& a, const matrix& b);       // a.rows() == b.rows() && a.cols() == b.cols()
    void check_dot(const matrix& a, const matrix& b);   // a.cols() == b.rows()

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
        float  *d;

        // -------------------------------------------------------------------
        // refcount

        void add_ref() const { p->refc++; }
        void release() const { if (0 == --p->refc) delete p; }

        // -------------------------------------------------------------------
        // operations

        void alloc(size_t n);
        void fill(float s);             // fill with s
        void fill(const data_p& that);  // copy the content
        void init(const data_p& that);

        // -------------------------------------------------------------------
        // properties

        [[nodiscard]] inline size_t size() const { return p->size; }
        [[nodiscard]] inline float* data() const { return d; }

        // -------------------------------------------------------------------
        // accessors

        // -------------------------------------------------------------------
        // assignments element-wise

        void assign(const data_p& that);            // assign by reference count

        void apply_eq(float (*fun)(float));
        void apply_eq(float (*fun)(float, float), float s);
        void apply_eq(float (*fun)(float, float), const data_p& that);
        void apply_eq(float (*fun)(float, float, float), float s, const data_p& that);
    };

    // ----------------------------------------------------------------------
    // vector
    // ----------------------------------------------------------------------

    struct vector;
    struct matrix;

    // ----------------------------------------------------------------------
    // vector
    // ----------------------------------------------------------------------

    struct vector: public data_p {

        // -------------------------------------------------------------------
        // constructor & destructor

        vector();
        explicit vector(size_t n);
        vector(float s, size_t n);
        vector(const vector& that, bool clone=false);
        ~vector() { release(); }

        [[nodiscard]] vector clone() const { return vector{self, true}; }

        // -------------------------------------------------------------------
        // access

        [[nodiscard]] inline float& at(size_t i)       { return p->data[i]; }
        [[nodiscard]] inline float  at(size_t i) const { return p->data[i]; }

        float& operator[](size_t i)       { return at(i); }
        float  operator[](size_t i) const { return at(i); }

        // -------------------------------------------------------------------
        // assignment

        vector& operator  = (float s) { fill(s);   return self; }
        vector& operator += (float s) { apply_eq(add, s); return self; }
        vector& operator -= (float s) { apply_eq(sub, s); return self; }
        vector& operator *= (float s) { apply_eq(mul, s); return self; }
        vector& operator /= (float s) { apply_eq(div, s); return self; }

        vector& operator  = (const vector& v) { assign(v); return self; }
        vector& operator += (const vector& v) { apply_eq(add, v); return self; }
        vector& operator -= (const vector& v) { apply_eq(sub, v); return self; }
        vector& operator *= (const vector& v) { apply_eq(mul, v); return self; }
        vector& operator /= (const vector& v) { apply_eq(div, v); return self; }

        vector& linear_eq (float a, const vector& v) { apply_eq(lin, a, v); return self; }

        // -------------------------------------------------------------------
        // operators

        vector operator +() const { vector r(self); return r; }
        vector operator -() const { vector r(self, true); r.apply_eq(neg); return r; }

        vector  operator +(const vector& v) const { vector r(self, true); r.apply_eq(add, v); return r; }
        vector  operator -(const vector& v) const { vector r(self, true); r.apply_eq(sub, v); return r; }
        vector  operator *(const vector& v) const { vector r(self, true); r.apply_eq(mul, v); return r; }
        vector  operator /(const vector& v) const { vector r(self, true); r.apply_eq(div, v); return r; }

        [[nodiscard]] vector linear(float s, const vector& that) const {
            vector r(self, true);
            r.apply_eq(lin, s, that);
            return self;
        }

        [[nodiscard]] float  dot(const vector& v) const;
        [[nodiscard]] vector dot(const matrix& m) const;
        [[nodiscard]] matrix cross(const vector& v) const;

        // -------------------------------------------------------------------
        // other

        [[nodiscard]] const vector& print() const;
    };

    // ----------------------------------------------------------------------

    inline vector zeros(size_t n) { return vector{0, n}; }
    inline vector  ones(size_t n) { return vector{1, n}; }
    vector range(size_t n);

    // ----------------------------------------------------------------------
    // operators

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

        // ----------------------------------------------------------------------
        // constructor

        matrix();
        explicit matrix(size_t n);
        matrix(size_t n, size_t m, float s=0);
        matrix(const matrix& m, bool clone=false);

        [[nodiscard]] matrix clone() const { return matrix(self, true); }
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

        // ----------------------------------------------------------------------
        // assignment

        matrix& operator  = (float s) { fill(s);   return self; }
        matrix& operator += (float s) { apply_eq(add, s); return self; }
        matrix& operator -= (float s) { apply_eq(sub, s); return self; }
        matrix& operator *= (float s) { apply_eq(mul, s); return self; }
        matrix& operator /= (float s) { apply_eq(div, s); return self; }

        matrix& operator  = (const matrix& m);
        matrix& operator += (const matrix& m) { apply_eq(add, m); return self;}
        matrix& operator -= (const matrix& m) { apply_eq(sub, m); return self;}
        matrix& operator *= (const matrix& m) { apply_eq(mul, m); return self;}
        matrix& operator /= (const matrix& m) { apply_eq(div, m); return self;}

        matrix& linear_eq (float a, const matrix& m) { apply_eq(lin, a, m); return self;}

        // -------------------------------------------------------------------
        // operations

        matrix operator +()                 const { matrix r(self);              return r;}
        matrix operator -()                 const { matrix r(self, true); r.apply_eq(neg); return r; }

        matrix  operator +(const matrix& m) const { matrix r(self, true); r.apply_eq(add, m); return r; }
        matrix  operator -(const matrix& m) const { matrix r(self, true); r.apply_eq(sub, m); return r; }
        matrix  operator *(const matrix& m) const { matrix r(self, true); r.apply_eq(mul, m); return r; }
        matrix  operator /(const matrix& m) const { matrix r(self, true); r.apply_eq(div, m); return r; }

        [[nodiscard]] matrix linear(float a, const matrix& m) const {
            matrix r(self, true);
            r.apply_eq(lin, a, m);
            return r;
        }

        [[nodiscard]] matrix dot(const matrix& m) const;
        [[nodiscard]] vector dot(const vector& v) const;

        // -------------------------------------------------------------------
        // other

        [[nodiscard]] const matrix& print() const;
    };

    // ----------------------------------------------------------------------
    // constructors

    matrix identity(size_t n);
    matrix zeros(size_t n, size_t m) { return matrix{n, m, 0}; }
    matrix  ones(size_t n, size_t m) { return matrix{n, m, 1}; }
    matrix range(size_t n, size_t m);

    // ----------------------------------------------------------------------
    // operators

}

#endif //CHECK_LINALG_LINALG_H
