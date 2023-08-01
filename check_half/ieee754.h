//
// Created by Corrado Mio on 24/07/2023.
//

#ifndef CHECK_HALF_IEEE754_H
#define CHECK_HALF_IEEE754_H

#include <cstdint>

/*
 * inf + n   = inf
 * n + inf   = inf
 * inf + inf = inf
 *
 * inf - n   = inf
 * n - inf   = -inf
 * inf - inf = nan
 *
 * inf * n   = inf
 * n * inf   = inf
 * inf * inf = inf
 * inf * 0   = nan
 *
 * inf / n   = inf
 * n / inf   = 0
 * inf / inf = nan
 * inf / 0   = inf
 * inf / n   = inf
 * 0 / inf   = 0
 * 0 / 0     = nan
 *
 * if an operand is nan, the result is nan
 */

namespace ieee754 {

    /*
     * float32
     */
    union float32;

    extern float32 add(const float32& f1, const float32& f2);
    extern float32 sub(const float32& f1, const float32& f2);
    extern float32 mul(const float32& f1, const float32& f2);
    extern float32 div(const float32& f1, const float32& f2);
    extern int cmp(const float32& f1, const float32& f2);

    static constexpr uint32_t E_LEN = 8;
    static constexpr uint32_t M_LEN = 23;
    static constexpr uint32_t E_OFS = (1 << (E_LEN - 1)) - 1;
    static constexpr uint32_t E_INF = (1 << E_LEN) - 1;
    static constexpr uint32_t M_MAX = (1 << M_LEN);
    static constexpr uint32_t M_TOP = (M_MAX << 1);
    static constexpr uint32_t M_BTM = (M_MAX >> 1);

    union float32 {

        // static constexpr uint32_t E_LEN = 8;
        // static constexpr uint32_t M_LEN = 23;
        // static constexpr uint32_t E_OFS = (1 << (E_LEN - 1)) - 1;
        // static constexpr uint32_t E_INF = (1 << E_LEN) - 1;
        static constexpr uint32_t M_MAX = (1 << M_LEN);
        // static constexpr uint32_t M_TOP = (M_MAX << 1);
        // static constexpr uint32_t M_BTM = (M_MAX >> 1);

        struct { uint32_t m: 23; uint32_t e: 8; uint32_t s:1; };
        uint32_t data;
        float value;

        float32() { }
        float32(int8_t s, uint16_t e, uint32_t m): s(s),e(e),m(m) { }
        explicit float32(float f): data(*(long*)(&f)){ }
        float32(const float32& f): data(f.data) {}

        operator float() const {
            return value;
        }

        float32 operator-(void) const {
            float32 r(*this);
            r.s ^= 1;
            return r;
        }

        inline float32& operator =(float f) {
            value = f;
            return *this;
        }

        inline float32& operator =(const float32& f) {
            data = f.data;
            return *this;
        }

        inline bool operator==(const float32& f) const {
            return cmp(*this, f) == 0;
        }

        inline bool operator<=(const float32& f) const {
            return cmp(*this, f) <= 0;
        }

        inline bool operator<(const float32& f) const {
            return cmp(*this, f) < 0;
        }

        inline bool operator!=(const float32& f) const {
            return cmp(*this, f) != 0;
        }

        inline bool operator>=(const float32& f) const {
            return cmp(*this, f) >= 0;
        }

        inline bool operator>(const float32& f) const {
            return cmp(*this, f) > 0;
        }

    };

    inline float32 operator +(const float32& f1, const float32& f2) {
        return (f1.s == f2.s) ? add(f1, f2) : sub(f1, f2);
    }

    inline float32 operator -(const float32& f1, const float32& f2) {
        return (f1.s == f2.s) ? sub(f1, f2) : add(f1, f2);
    }

    inline float32 operator *(const float32& f1, const float32& f2) {
        return mul(f1, f2);
    }

    inline float32 operator /(const float32& f1, const float32& f2) {
        return div(f1, f2);
    }

}

#endif //CHECK_HALF_IEEE754_H
