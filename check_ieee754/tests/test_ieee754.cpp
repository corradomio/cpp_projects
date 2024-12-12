//
// Created by Corrado Mio on 02/03/2024.
//
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include "std/real_limits.h"

using namespace std;
using namespace ieee754;


TEST_CASE( "constructor", "[float32_t]" ) {
    float32_t f32(0.);

    REQUIRE(float(f32) == 0);
    REQUIRE(numeric_limits<float32_t>::min() == numeric_limits<float>::min());
    REQUIRE(numeric_limits<float32_t>::max() == numeric_limits<float>::max());
    REQUIRE(numeric_limits<float32_t>::epsilon() == numeric_limits<float>::epsilon());
    REQUIRE(numeric_limits<float32_t>::infinity() == numeric_limits<float>::infinity());
    REQUIRE(numeric_limits<float32_t>::denorm_min() == numeric_limits<float>::denorm_min());

    REQUIRE(numeric_limits<float32_t>::radix == numeric_limits<float>::radix);
    REQUIRE(numeric_limits<float32_t>::digits == numeric_limits<float>::digits);
    REQUIRE(numeric_limits<float32_t>::min_exponent == numeric_limits<float>::min_exponent);
    REQUIRE(numeric_limits<float32_t>::max_exponent == numeric_limits<float>::max_exponent);

    REQUIRE(numeric_limits<float32_t>::is_specialized == numeric_limits<float>::is_specialized);
    REQUIRE(numeric_limits<float32_t>::is_signed == numeric_limits<float>::is_signed);
    REQUIRE(numeric_limits<float32_t>::is_integer == numeric_limits<float>::is_integer);
    REQUIRE(numeric_limits<float32_t>::is_exact == numeric_limits<float>::is_exact);
    REQUIRE(numeric_limits<float32_t>::has_infinity == numeric_limits<float>::has_infinity);
    REQUIRE(numeric_limits<float32_t>::has_quiet_NaN == numeric_limits<float>::has_quiet_NaN);
    REQUIRE(numeric_limits<float32_t>::has_signaling_NaN == numeric_limits<float>::has_signaling_NaN);
    REQUIRE(numeric_limits<float32_t>::has_denorm == numeric_limits<float>::has_denorm);
    REQUIRE(numeric_limits<float32_t>::has_denorm_loss == numeric_limits<float>::has_denorm_loss);
    REQUIRE(numeric_limits<float32_t>::is_iec559 == numeric_limits<float>::is_iec559);
    REQUIRE(numeric_limits<float32_t>::is_bounded == numeric_limits<float>::is_bounded);
    REQUIRE(numeric_limits<float32_t>::is_modulo == numeric_limits<float>::is_modulo);

}


TEST_CASE( "constructor", "[float64_t]" ) {
    float64_t f32(0.);

    REQUIRE(double(f32 == 0.));
    REQUIRE(numeric_limits<float64_t>::min() == numeric_limits<double>::min());
    REQUIRE(numeric_limits<float64_t>::max() == numeric_limits<double>::max());
    REQUIRE(numeric_limits<float64_t>::epsilon() == numeric_limits<double>::epsilon());
    REQUIRE(numeric_limits<float64_t>::infinity() == numeric_limits<double>::infinity());
    REQUIRE(numeric_limits<float64_t>::denorm_min() == numeric_limits<double>::denorm_min());

    REQUIRE(numeric_limits<float64_t>::radix == numeric_limits<double>::radix);
    REQUIRE(numeric_limits<float64_t>::digits == numeric_limits<double>::digits);
    REQUIRE(numeric_limits<float64_t>::min_exponent == numeric_limits<double>::min_exponent);
    REQUIRE(numeric_limits<float64_t>::max_exponent == numeric_limits<double>::max_exponent);

    REQUIRE(numeric_limits<float64_t>::is_specialized == numeric_limits<double>::is_specialized);
    REQUIRE(numeric_limits<float64_t>::is_signed == numeric_limits<double>::is_signed);
    REQUIRE(numeric_limits<float64_t>::is_integer == numeric_limits<double>::is_integer);
    REQUIRE(numeric_limits<float64_t>::is_exact == numeric_limits<double>::is_exact);
    REQUIRE(numeric_limits<float64_t>::has_infinity == numeric_limits<double>::has_infinity);
    REQUIRE(numeric_limits<float64_t>::has_quiet_NaN == numeric_limits<double>::has_quiet_NaN);
    REQUIRE(numeric_limits<float64_t>::has_signaling_NaN == numeric_limits<double>::has_signaling_NaN);
    REQUIRE(numeric_limits<float64_t>::has_denorm == numeric_limits<double>::has_denorm);
    REQUIRE(numeric_limits<float64_t>::has_denorm_loss == numeric_limits<double>::has_denorm_loss);
    REQUIRE(numeric_limits<float64_t>::is_iec559 == numeric_limits<double>::is_iec559);
    REQUIRE(numeric_limits<float64_t>::is_bounded == numeric_limits<double>::is_bounded);
    REQUIRE(numeric_limits<float64_t>::is_modulo == numeric_limits<double>::is_modulo);

}


TEST_CASE( "number", "[float32_t]" ) {
    int elen = float32_t::ELEN;
    int mlen = float32_t::MLEN;

    float32_t f32;
    float64_t f64;
    float32_t g32;

    for (int e=0; e<elen; ++e) {
        for (int m = 0; m < mlen; ++m) {
            f32 = float32_t(0, 1 << e, 1 << m);
            f64 = f32;
            g32 = f64;

            REQUIRE(f32 == g32);
        }
    }

    // subnormal
    f32 = float32_t(0,0,1);
    f64 = f32;
    g32 = f64;
    REQUIRE(f32 == g32);

    // normal
    f32 = float32_t(0,1,0);
    f64 = f32;
    g32 = f64;
    REQUIRE(f32 == g32);

    // signed zero
    f32 = float32_t(1,0,0);
    f64 = f32;
    g32 = f64;
    REQUIRE(f32 == g32);

    // signed normal
    f32 = float32_t(1,1,1);
    f64 = f32;
    g32 = f64;
    REQUIRE(f32 == g32);

    // +inf
    f32 = float32_t(0,-1,0);
    f64 = f32;
    g32 = f64;
    REQUIRE(f32 == g32);

    // -inf
    f32 = float32_t(1,-1,0);
    f64 = f32;
    g32 = f64;
    REQUIRE(f32 == g32);

}

TEST_CASE( "fpclassify", "[float32_t]" ) {
    REQUIRE(fpclassify(float32_t(0,0,0)) == FP_ZERO);
    REQUIRE(fpclassify(float32_t(0,0,1)) == FP_SUBNORMAL);
    REQUIRE(fpclassify(float32_t(0,1,0)) == FP_NORMAL);
    REQUIRE(fpclassify(float32_t(0,-1,0)) == FP_INFINITE);
    REQUIRE(fpclassify(float32_t(0,-1,1)) == FP_NAN);
}

TEST_CASE( "fpclassify", "[float64_t]" ) {
    REQUIRE(fpclassify(float64_t(0,0,0)) == FP_ZERO);
    REQUIRE(fpclassify(float64_t(0,0,1)) == FP_SUBNORMAL);
    REQUIRE(fpclassify(float64_t(0,1,0)) == FP_NORMAL);
    REQUIRE(fpclassify(float64_t(0,-1,0)) == FP_INFINITE);
    REQUIRE(fpclassify(float64_t(0,-1,1)) == FP_NAN);
}

TEST_CASE( "float conversions", "[float32_t]" ) {
    REQUIRE(float(float32_t(128.16f)) == float32_u(128.16f).flt);
    REQUIRE(double(float32_t(128.)) == float64_u(128.).dbl);

    REQUIRE(float32_u(128.16f).flt == float(float32_t(128.16f)));
    REQUIRE(float64_u(128.).dbl == double(float32_t(128.)));
}

TEST_CASE( "float conversions", "[float64_t]" ) {
    REQUIRE(float(float64_t(128.16f)) == float32_u(128.16f).flt);
    REQUIRE(double(float64_t(128.16)) == float64_u(128.16).dbl);

    REQUIRE(float32_u(128.16f).flt == float(float64_t(128.16f)));
    REQUIRE(float64_u(128.16).dbl == double(float64_t(128.16)));
}

TEST_CASE( "double subnormal", "[float64_t]" ) {
    float64_t f64(0,0,-1);
    float32_t f32{f64};

    REQUIRE(float(f32) == 0.);
}

// unsupported  [float16_t]
// unsupported  [float128_t]

