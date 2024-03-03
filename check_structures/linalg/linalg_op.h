//
// Created by Corrado Mio on 27/02/2024.
//

#ifndef LINALLG_OP_H
#define LINALLG_OP_H

#include "linalg.h"

namespace stdx::linalg {

    // ----------------------------------------------------------------------
    // vector

    vector operator +(float s, const vector& v);
    vector operator -(float s, const vector& v);
    vector operator *(float s, const vector& v);
    vector operator /(float s, const vector& v);

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

    matrix operator +(float s, const matrix& m);
    matrix operator -(float s, const matrix& m);
    matrix operator *(float s, const matrix& m);
    matrix operator /(float s, const matrix& m);

}

#endif //LINALLG_OP_H
