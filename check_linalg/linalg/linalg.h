//
// Created by Corrado Mio on 14/02/2024.
//

#ifndef CHECK_LINALG_LINALG_H
#define CHECK_LINALG_LINALG_H

#include <cstddef>
#include <cmath>
#include <exception>
#include <stdexcept>

#define self (*this)

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

        // -------------------------------------------------------------------
        // refcount

        void add_ref() const { p->refc++; }
        void release() const { if (0 == --p->refc) delete p; }

        // -------------------------------------------------------------------
        // operations

        void alloc(size_t n);
        void init(float s);             // fill with s
        void init(const data_p& o);     // copy the content

        // -------------------------------------------------------------------
        // properties

        [[nodiscard]] inline size_t size() const { return p->size; }
        [[nodiscard]] inline float* data() const { return p->data; }

        // -------------------------------------------------------------------
        // accessors

        [[nodiscard]] inline float& at(size_t i)       { return p->data[i]; }   // write
        [[nodiscard]] inline float  at(size_t i) const { return p->data[i]; }   // read

        // -------------------------------------------------------------------
        // assignments element-wise

        void assign(const data_p& o);           // assign by reference count
        void add_eq(float s);                   // +=
        void sub_eq(float s);                   // -=
        void mul_eq(float s);                   // *=
        void div_eq(float s);                   // /=
        void neg_eq();                          // *= (-1)

        void add_eq(const data_p& v);           // +=
        void sub_eq(const data_p& v);           // -=
        void mul_eq(const data_p& v);           // *=
        void div_eq(const data_p& v);           // /=
        void lin_eq(float a, const data_p& o);  // += a*o

        void apply_eq(float (*f)(float));
        void apply_eq(float (*f)(float, float), float b);
        void apply_eq(float (*f)(float, float), const data_p& o);
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
        vector(const vector& v, bool clone=false);
        ~vector() { release(); }

        [[nodiscard]] vector clone() const { return vector(self, true); }

        // -------------------------------------------------------------------
        // access

        [[nodiscard]] inline float& at(size_t i)       { return p->data[i]; }
        [[nodiscard]] inline float  at(size_t i) const { return p->data[i]; }

        float& operator[](size_t i)       { return at(i); }
        float  operator[](size_t i) const { return at(i); }

        // -------------------------------------------------------------------
        // assignment

        vector& operator  = (float s) { init(s);   return self; }
        vector& operator += (float s) { add_eq(s); return self; }
        vector& operator -= (float s) { sub_eq(s); return self; }
        vector& operator *= (float s) { mul_eq(s); return self; }
        vector& operator /= (float s) { div_eq(s); return self; }

        vector& operator  = (const vector& v) { assign(v); return self; }
        vector& operator += (const vector& v) { add_eq(v); return self; }
        vector& operator -= (const vector& v) { sub_eq(v); return self; }
        vector& operator *= (const vector& v) { mul_eq(v); return self; }
        vector& operator /= (const vector& v) { div_eq(v); return self; }

        vector& linear_eq (float a, const vector& v) { lin_eq(a, v); return self; }

        // -------------------------------------------------------------------
        // operators

        vector operator +() const { vector r(self); return r; }
        vector operator -() const { vector r(self, true); r.neg_eq(); return r; }

        vector  operator +(const vector& v) const { vector r(self, true); r.add_eq(v); return r; }
        vector  operator -(const vector& v) const { vector r(self, true); r.sub_eq(v); return r; }
        vector  operator *(const vector& v) const { vector r(self, true); r.mul_eq(v); return r; }
        vector  operator /(const vector& v) const { vector r(self, true); r.div_eq(v); return r; }

        [[nodiscard]] vector linear(float a, const vector& v) const {
            vector r(self, true);
            r.lin_eq(a, v);
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

    vector operator +(float s, const vector& v);
    vector operator -(float s, const vector& v);
    vector operator *(float s, const vector& v);
    vector operator /(float s, const vector& v);

    // ----------------------------------------------------------------------
    // functions

    vector abs(const vector& v);
    vector log(const vector& v);
    vector exp(const vector& v);
    vector pow(const vector& v, float e);
    vector sq( const vector& v);
    vector sqrt(const vector& v);

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

        // ----------------------------------------------------------------------
        // constructor

        matrix();
        explicit matrix(size_t n);
        matrix(size_t n, size_t m);
        matrix(float s, size_t n, size_t m);
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
        matrix operator -()                 const { matrix r(self, true); r.neg_eq( ); return r; }

        matrix  operator +(const matrix& m) const { matrix r(self, true); r.add_eq(m); return r; }
        matrix  operator -(const matrix& m) const { matrix r(self, true); r.sub_eq(m); return r; }
        matrix  operator *(const matrix& m) const { matrix r(self, true); r.mul_eq(m); return r; }
        matrix  operator /(const matrix& m) const { matrix r(self, true); r.div_eq(m); return r; }

        [[nodiscard]] matrix linear(float a, const matrix& m) const {
            matrix r(self, true);
            r.lin_eq(a, m);
            return r;
        }

        [[nodiscard]] matrix dot(const matrix& m) const;
        [[nodiscard]] vector dot(const vector& v) const;

        // -------------------------------------------------------------------
        // other

        [[nodiscard]] const matrix& print() const;
    };

    // ----------------------------------------------------------------------

    matrix identity(size_t n);
    inline matrix zeros(size_t n, size_t m) { return matrix{0, n, m}; }
    inline matrix  ones(size_t n, size_t m) { return matrix{1, n, m}; }
    matrix range(size_t n, size_t m);

    // ----------------------------------------------------------------------
    // operators

    matrix operator +(float s, const matrix& m);
    matrix operator -(float s, const matrix& m);
    matrix operator *(float s, const matrix& m);
    matrix operator /(float s, const matrix& m);

}

#endif //CHECK_LINALG_LINALG_H
