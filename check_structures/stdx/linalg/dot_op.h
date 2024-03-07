//
// Created by Corrado Mio on 03/03/2024.
//

#ifndef STDX_DOT_OP_H
#define STDX_DOT_OP_H

#include "vector.h"
#include "matrix.h"

//
// Check for dot operation
//

namespace stdx::linalg {

    template<typename T>
    void check_dot(const vector_t<T> &u, const vector_t<T> &v) {
        if (u.size() != v.size())
            throw bad_dimensions();
    }

    template<typename T>
    void check_dot(const matrix_t<T> &m, const vector_t<T> &v) {
        if (m.cols() != v.size())
            throw bad_dimensions();
    }

    template<typename T>
    void check_dot(const vector_t<T> &v, const matrix_t<T> &m) {
        if (v.size() != m.rows())
            throw bad_dimensions();
    }

    template<typename T>
    void check_dot(const matrix_t<T> &m1, const matrix_t<T> &m2) {
        if (m1.cols() != m2.rows())
            throw bad_dimensions();
    }

}


//
// vector_t
//
//      dot(u, v)
//

namespace stdx::linalg {

    template<typename T>
    T dot(const vector_t<T>& v1, const vector_t<T>& v2) {
        check_dot(v1, v2);
        T s = 0;
        size_t n = v1.size();
        T* x = v1.data();
        T* y = v2.data();
        for (int i=0; i<n; ++i)
            s += x[i]*y[i];
        return s;
    }

}


//
// matrix_t/vector_t
//
//      dot(m, m)
//      dot(m, v)
//      dot(v, m)
//

namespace stdx::linalg {

    template<typename T>
    vector_t<T> dot(const matrix_t<T>& m, const vector_t<T>& v) {
        check_dot(m, v);
        size_t nr = m.rows();
        size_t nc = m.cols();
        vector_t<T> r(m.rows());

        for(size_t i=0,k=0; i<nr; ++i,k+=nc) {
            T s = 0;
            for (size_t j=0; j<nc; ++j)
                s += v[j]*m[k];

            r[i] = s;
        }
        return r;
    }


    template<typename T>
    vector_t<T> dot(const vector_t<T>& v, const matrix_t<T>& m) {
        check_dot(v, m);
        size_t nr = m.rows();
        size_t nc = m.cols();
        vector_t<T> r(m.cols());

        for(size_t i=0; i<nc; ++i) {
            T s = 0;
            for (size_t j=0,k=i; j<nr; ++j,k+=nc)
                s += v[j]*m[k];

            r[i] = s;
        }
        return r;
    }


    template<typename T>
    matrix_t<T> dot(const matrix_t<T>& m1, const matrix_t<T>& m2) {
        check_dot(m1, m2);
        size_t nr = m1.rows();
        size_t nc = m2.cols();
        size_t nh = m1.cols();
        matrix_t<T> r(nr, nc);

        for(size_t i=0; i<nr; ++i) {
            for (size_t j=0; j<nc; ++j) {
                T s = 0;
                for(size_t h=0,k=j; h<nh; ++h,k+=nc)
                    s += m1[i,h]*m2[k];

                r[i,j] = s;
            }
        }
        return r;
    }

}


//
// vector_t dot methods
//
//      u.dot(v)
//      u.dot(m)
//

namespace stdx::linalg {

    template<typename T>
    T vector_t<T>::dot(const vector_t& b) const {
        return stdx::linalg::dot(self, b);
    }

    template<typename T>
    vector_t<T> vector_t<T>::dot(const matrix_t<T>& m) const {
        return stdx::linalg::dot(self, m);
    }

}



//
// matrix_t dot methods
//
//      m.dot(m)
//      m.dot(v)
//

namespace stdx::linalg {

    template<typename T>
    vector_t<T> matrix_t<T>::dot(const vector_t<T>& b) const {
        return stdx::linalg::dot(self, b);
    }

    template<typename T>
    matrix_t<T> matrix_t<T>::dot(const matrix_t& m) const {
        return stdx::linalg::dot(self, m);
    }

}

#endif //STDX_DOT_OP_H
