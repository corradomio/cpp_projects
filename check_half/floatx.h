//
// Created by Corrado Mio on 22/07/2023.
//

#ifndef CHECK_HALF_FLOATX_H
#define CHECK_HALF_FLOATX_H

// char         1
// short        2
// long         4
// long long    8
// float        4
// double       8

namespace numeric {

    struct float16_t {
        short i16;
    };

    struct bfloat16_t {
        short i16;
    };

    union float32_t {
        float r32; long i32;

        inline float32_t() {}
        inline float32_t(float f) { r32=f; }
        inline float32_t(const float32_t& f) { r32 = f.r32; }

        inline float32_t& operator  =(const float32_t& f) { r32 =f.r32; return *this; }
        inline float32_t& operator +=(const float32_t& f) { r32+=f.r32; return *this; }
        inline float32_t& operator -=(const float32_t& f) { r32-=f.r32; return *this; }
        inline float32_t& operator *=(const float32_t& f) { r32*=f.r32; return *this; }
        inline float32_t& operator /=(const float32_t& f) { r32/=f.r32; return *this; }

        inline float32_t  operator + (const float32_t& f) const { float32_t r{*this}; return r.r32+=f.r32; }
        inline float32_t  operator - (const float32_t& f) const { float32_t r{*this}; return r.r32-=f.r32; }
        inline float32_t  operator * (const float32_t& f) const { float32_t r{*this}; return r.r32*=f.r32; }
        inline float32_t  operator / (const float32_t& f) const { float32_t r{*this}; return r.r32/=f.r32; }

        inline float32_t  operator + (float f) const { float32_t r{*this}; return r.r32+=f; }
        inline float32_t  operator - (float f) const { float32_t r{*this}; return r.r32-=f; }
        inline float32_t  operator * (float f) const { float32_t r{*this}; return r.r32*=f; }
        inline float32_t  operator / (float f) const { float32_t r{*this}; return r.r32/=f; }

        inline operator float() const { return r32; }

        inline static float32_t from_bits(long i32) {
            float32_t r;
            r.i32 = i32;
            return r;
        }

        inline long to_bits() const { return i32; }

    };

    struct float64_t {
        union { double r64; long long i64; } value;
    };

}

#endif //CHECK_HALF_FLOATX_H
