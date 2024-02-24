//
// Created by Corrado Mio on 07/02/2024.
//

#ifndef CHECK_NBODY_V3D_H
#define CHECK_NBODY_V3D_H

#include "array.h"

#define self (*this)


namespace space {

    struct v3d {
        float x, y, z;

        explicit v3d() : x(0), y(0), z(0) {}

        v3d(float x, float y, float z) : x(x), y(y), z(z) {}

        // v3d(const v3d& v): x(v.x), y(v.y), z(v.z) { }
        v3d(const v3d &v) = default;

        // v3d& operator =(const v3d& v) {
        //     x = v.x;
        //     y = v.y;
        //     z = v.z;
        //     return self;
        // }
        v3d &operator=(const v3d &v) = default;

        v3d &operator=(float v) {
            x = v;
            y = v;
            z = v;
            return self;
        }

        v3d &operator+=(const v3d &v) {
            x += v.x;
            y += v.y;
            z += v.z;
            return self;
        }

        v3d &operator-=(const v3d &v) {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return self;
        }

        v3d &operator*=(const v3d &v) {
            x *= v.x;
            y *= v.y;
            z *= v.z;
            return self;
        }

        v3d &operator*=(float s) {
            x *= s;
            y *= s;
            z *= s;
            return self;
        }

        v3d &operator/=(const v3d &v) {
            x /= v.x;
            y /= v.y;
            z /= v.z;
            return self;
        }

        v3d &operator/=(float s) {
            x /= s;
            y /= s;
            z /= s;
            return self;
        }

        v3d &linear(float s, const v3d &v) {
            x += s * v.x;
            y += s * v.y;
            z += s * v.z;
            return self;
        }

    };


};

#endif //CHECK_NBODY_V3D_H
