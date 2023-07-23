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

//
// FP64    E11F53  s eeeeeeeeeee mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// FP32    E8M23   s eeeeeeee    mmmmmmmmmmmmmmmmmmmmmmm
// FP24    E8M15   s eeeeeeee    mmmmmmmmmmmmmmm
// FP16    E5M10   s eeeee       mmmmmmmmmm
// BF16    E8M7    s eeeeeeee    mmmmmmm
// FP8     E4M3    s eeee        mmm
//         E5M2    s eeeee       mm
// TF32    E8M10   s eeeeeeee    mmmmmmmmmm

namespace numeric {

    // E5M10
    struct float16_t {
        char c[2];
        struct { unsigned m: 10; unsigned e:5; unsigned s: 1; } parts;
        short i16;
    };

    // E8M7
    struct bfloat16_t {
        char c[2];
        struct { unsigned m: 7; unsigned e:8; unsigned s: 1; } parts;
        short i16;
    };

    // E11F53
    union float64_t {
        char c[8];
        struct { unsigned long long m: 53; unsigned e:11; unsigned s: 1; } parts;
        float r64;
        long long i64;
    };

    // E8M23
    union float32_t {
        char c[4];
        struct { unsigned m: 23; unsigned e:8; unsigned s: 1; } parts;
        float r32;
        long  i32;

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

}

#endif //CHECK_HALF_FLOATX_H
