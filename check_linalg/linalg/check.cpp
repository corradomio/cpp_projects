//
// Created by Corrado Mio on 24/02/2024.
//
#include "linalg.h"

using namespace stdx::linalg;


void stdx::linalg::check(const data_p& u, const data_p& v) {
    if (u.size() != v.size())
        throw bad_dimensions();
}

void stdx::linalg::check(const vector& u, const vector& v) {
    if (u.size() != v.size())
        throw bad_dimensions();
}


void stdx::linalg::check(const matrix& a, const matrix& b) {
    if (a.size() != b.size())
        throw bad_dimensions();
    if (a.c != b.c)
        throw bad_dimensions();
}


void stdx::linalg::check(const matrix& m, const vector& v) {
    if (m.cols() != v.size())
        throw bad_dimensions();
}


void stdx::linalg::check(const vector& u, const matrix& m) {
    if (u.size() != m.rows())
        throw bad_dimensions();
}


void stdx::linalg::check_dot(const matrix& a, const matrix& b) {
    if (a.cols() != b.rows())
        throw bad_dimensions();
}
