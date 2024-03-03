//
// Created by Corrado Mio on 27/02/2024.
//
#include "linalg_op.h"

using namespace stdx::linalg;


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
