//
// Created by Corrado Mio on 11/02/2024.
//

#ifndef CHECK_STRUCTURES_V3D_H
#define CHECK_STRUCTURES_V3D_H

namespace stdx {

    struct v3d_t {
        float x,y,z;

        v3d_t(): x(0),y(0),z(0) {}
        v3d_t(float x, float y, float z): x(x), y(y), z(z) {}
        v3d_t(const v3d_t& v) = default;

        v3d_t& operator =(float v) {
            x = y = z = v;
            return *this;
        }
        v3d_t& operator =(const v3d_t& v) = default;

        v3d_t& operator +=(const v3d_t& v) {
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }
        v3d_t& operator -=(const v3d_t& v) {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return *this;
        }
        v3d_t& operator *=(const v3d_t& v) {
            x *= v.x;
            y *= v.y;
            z *= v.z;
            return *this;
        }
        v3d_t& operator *=(float v) {
            x *= v;
            y *= v;
            z *= v;
            return *this;
        }
        v3d_t& operator /=(const v3d_t& v) {
            x /= v.x;
            y /= v.y;
            z /= v.z;
            return *this;
        }
        v3d_t& operator /=(float v) {
            x /= v;
            y /= v;
            z /= v;
            return *this;
        }
        v3d_t operator +(const v3d_t& v) const {
            v3d_t r(*this);
            r += v;
            return r;
        }
        v3d_t operator -(const v3d_t& v) const {
            v3d_t r(*this);
            r -= v;
            return r;
        }
        v3d_t operator *(float v) const {
            v3d_t r(*this);
            r *= v;
            return r;
        }
        v3d_t operator /(float v) const {
            v3d_t r(*this);
            r /= v;
            return r;
        }
    };

    inline v3d_t operator *(float s, const v3d_t& v) {
        return v*s;
    }
}

#endif //CHECK_STRUCTURES_V3D_H
