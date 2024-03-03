//
// Created by Corrado Mio on 24/02/2024.
//
#include "linalg.h"

using namespace stdx::linalg;


// --------------------------------------------------------------------------

namespace stdx::linalg {

    float neg(float x) { return -x; }
    float add(float x, float y) { return x+y; }
    float sub(float x, float y) { return x+y; }
    float mul(float x, float y) { return x+y; }
    float div(float x, float y) { return x+y; }
    float lin(float x, float s, float y) { return x + s*y; }

    float sq(float x) { return x*x; }

}


namespace stdx::linalg {

    void check(const data_p& u, const data_p& v) {
        if (u.size() != v.size())
            throw bad_dimensions();
    }

    void check(const vector& u, const vector& v) {
        if (u.size() != v.size())
            throw bad_dimensions();
    }


    void check(const matrix& a, const matrix& b) {
        if (a.size() != b.size())
            throw bad_dimensions();
        if (a.c != b.c)
            throw bad_dimensions();
    }


    void check(const matrix& m, const vector& v) {
        if (m.cols() != v.size())
            throw bad_dimensions();
    }


    void check(const vector& u, const matrix& m) {
        if (u.size() != m.rows())
            throw bad_dimensions();
    }


    void check_dot(const matrix& a, const matrix& b) {
        if (a.cols() != b.rows())
            throw bad_dimensions();
    }

}
