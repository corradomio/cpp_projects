//
// Created by Corrado Mio on 02/03/2024.
//

#include <limits>
#include "real_t.h"

#ifndef IEEE754_REAL_LIMITS_H
#define IEEE754_REAL_LIMITS_H


/*
 *  Numeric limits
 *  --------------
 *      Add support for 'std::numeric_limits<real<E,M>>'
 *
 *      numeric_limits<bool>
 *      numeric_limits<char>
 *      numeric_limits<signed char>
 *      numeric_limits<unsigned char>
 *      numeric_limits<wchar_t>
 *      numeric_limits<char16_t>
 *      numeric_limits<char32_t>
 *      numeric_limits<short>
 *      numeric_limits<unsigned short>
 *      numeric_limits<int>
 *      numeric_limits<unsigned int>
 *      numeric_limits<long>
 *      numeric_limits<unsigned long>
 *      numeric_limits<long long>
 *      numeric_limits<unsigned long long>
 *
 *      numeric_limits<float>
 *      numeric_limits<double>
 *      numeric_limits<long double>
 */

namespace std {

    using namespace ieee754;


    template<int E, int M, typename T>
    struct numeric_limits<real_t<E, M, T>> {
    typedef real_t<E, M, T> Float;

    static constexpr Float min() noexcept { return {0, 1, 0}; }
    static constexpr Float max() noexcept { return {0, typename Float::field_type(-2), typename Float::field_type(-1)}; }
    static constexpr Float epsilon() noexcept { return {0, typename Float::field_type(Float::EBIAS-Float::MLEN), 0}; }
    static constexpr Float lowest() noexcept { return {1, typename Float::field_type(-2), typename Float::field_type(-1)}; }
    static constexpr Float infinity() noexcept { return {0, typename Float::field_type(-1), 0}; }
    // static constexpr Float quiet_NaN() noexcept { return {0, typename Float::field_type(-1), 1}; }
    // static constexpr Float signaling_NaN() noexcept { return {1, typename Float::field_type(-1), 1}; }
    static constexpr Float denorm_min() noexcept { return {0, 0, 1}; }

    static constexpr int radix = 2;
    static constexpr int digits = int(Float::MLEN+1);
    static constexpr int min_exponent = 2 - int(Float::EBIAS);
    static constexpr int max_exponent = int(Float::EMAX - Float::EBIAS);

    static _GLIBCXX_USE_CONSTEXPR bool is_specialized = true;
    static _GLIBCXX_USE_CONSTEXPR bool is_signed = true;
    static _GLIBCXX_USE_CONSTEXPR bool is_integer = false;
    static _GLIBCXX_USE_CONSTEXPR bool is_exact = false;
    static _GLIBCXX_USE_CONSTEXPR bool has_infinity = true;
    static _GLIBCXX_USE_CONSTEXPR bool has_quiet_NaN = true;
    static _GLIBCXX_USE_CONSTEXPR bool has_signaling_NaN = has_quiet_NaN;
    static _GLIBCXX_USE_CONSTEXPR float_denorm_style has_denorm = denorm_present;
    static _GLIBCXX_USE_CONSTEXPR bool has_denorm_loss = false;
    static _GLIBCXX_USE_CONSTEXPR bool is_iec559 = has_infinity && has_quiet_NaN && has_denorm == denorm_present;
    static _GLIBCXX_USE_CONSTEXPR bool is_bounded = true;
    static _GLIBCXX_USE_CONSTEXPR bool is_modulo = false;
};

}

#endif //IEEE754_REAL_LIMITS_H
