#include <stdio.h>
#include <iostream>
#include <ieee754/real_t.h>
#include <std/real_limits.h>

using namespace ieee754;
using namespace std;


typedef real_t<3,4, uint8_t>  f34_t;
typedef real_t<2,5, uint8_t>  f25_t;
typedef real_t<4,3, uint8_t>  f43_t;


int main8() {
    float64_u f64, g64;

    f64.f64 = float64_t{0,float64_t::EBIAS, 0};
    g64.f64 = float64_t{0,float64_t::EBIAS, 1};

    std::cout << g64.dbl - f64.dbl << std::endl;

    return 0;
}


int main7() {
    float64_u  u;
    u.f64 = float64_t(0,0,-1);

    float32_u v;

    v.f32 = u.f64;
    v.flt = 1;

    u.f64 = float64_t(0,float64_t::EBIAS, 0);
    printf("%f\n", u.dbl);

    u.f64 = float64_t(0,float64_t::EBIAS+1, 0);
    printf("%f\n", u.dbl);

    return 0;
}


int main6(){
    float32_u f32, g32;
    float64_u f64, g64;

    // f32 = float32_t(0,-1,0);
    // f64 = f32;
    // g32 = f64;

    f32.f32 = float32_t(0,0,1);
    f64.f64 = f32.f32;
    g32.f32 = f64.f64;

    g64.dbl = 1.40129846e-045;

    f64.f64 = float64_t{0,float64_t::EBIAS, 0};
    g64.f64 = float64_t{0,float64_t::EBIAS, 1};

    std::cout << g64.dbl - f64.dbl << std::endl;

    return 0;
}

int main5() {
    float32_u u(128.e16f);
    float32_t v(128.e16f);

    float32_u f1; f1.flt = float(float32_t(128.e16f));
    float32_u f2; f2.flt = float32_u(128.e16f).flt;

    return 0;
}


int main4() {
    float16_t f(0,0,1);

    float16_u u;
    u.f16 = numeric_limits<float32_t>::min();

    float16_u v;
    v.flt = numeric_limits<__float16>::min();

    float32_t f32{1.};
    float64_t f64{1.};

    bool cmp = (f32 == f64);


    return 0;
}


int main3() {
    // float32_u v1;
    // float64_u v2;
    // float32_u v3;
    //
    // v1.f32 =  float32_t(0,0,1);
    // v2.f64 = v1.f32;
    // v3.f32 = v2.f64;
    //
    // std::cout << (double)v1.f32 << std::endl;
    // std::cout << (double)v2.f64 << std::endl;
    // std::cout << (double)v3.f32 << std::endl;

    // float32_t defval{0,0,1};
    // float64_t larger{defval};
    // float16_t smaller{defval};

    f34_t defval{0, 0, 1};
    f25_t larger{defval};
    f43_t smaller{defval};


    std::cout << (float)defval << std::endl;
    std::cout << (float)larger << std::endl;
    std::cout << (float)smaller << std::endl;

    return 0;
}


int main2() {
    // float64_t f(0,0,-1);
    //
    // std::cout << double(f) << std::endl;
    //
    // float32_t g(f);
    // std::cout << float(g) << std::endl;

    float32_t f(0,0,1);
    float64_t g{f};

    std::cout << float(f) << std::endl;
    std::cout << double(g) << std::endl;

    return 0;
}



int main() {
    int x = __FLT_RADIX__;
    float64_t f64;

    std::cout << "-- " << std::endl;
    std::cout << "sizeof(float) "      << sizeof(float) << std::endl;
    std::cout << "sizeof(double) "      << sizeof(double) << std::endl;
    std::cout << "sizeof(long double) " << sizeof(long double) << std::endl;
    std::cout << "sizeof(__float128) "  << sizeof(__float128) << std::endl;
    std::cout << "sizeof(__int128) "    << sizeof(__int128) << std::endl;
    std::cout << "-- " << std::endl;
    std::cout << "sizeof(float8_t) "    << sizeof(float8_t) << std::endl;
    std::cout << "sizeof(bfloat8_t) "   << sizeof(bfloat8_t) << std::endl;
    std::cout << "sizeof(float16_t) "   << sizeof(float16_t) << std::endl;
    std::cout << "sizeof(bfloat16_t) "  << sizeof(bfloat16_t) << std::endl;
    std::cout << "sizeof(float32_t) "   << sizeof(float32_t) << std::endl;
    std::cout << "sizeof(float64_t) "   << sizeof(float64_t) << std::endl;
    std::cout << "sizeof(float128_t) "  << sizeof(float128_t) << std::endl;



    // float128_u ldmin1; ldmin1.dbl  = std::numeric_limits<long double>::min();
    // float128_u ldmin2; ldmin2.f128 = std::numeric_limits<float128_t>::min();
    // float128_u ldmax1; ldmax1.dbl  = std::numeric_limits<long double>::max();
    // float128_u ldmax2; ldmax2.f128 = std::numeric_limits<float128_t>::max();
    // float128_u ldeps1; ldeps1.dbl  = std::numeric_limits<long double>::epsilon();
    // float128_u ldeps2; ldeps2.f128 = std::numeric_limits<float128_t>::epsilon();

    std::cout << "== numeric_limits == " << std::endl;
    std::cout << "-- long double" << std::endl;
    std::cout << (long double)std::numeric_limits<long double>::min() << std::endl;
    std::cout << (long double)std::numeric_limits<long double>::epsilon() << std::endl;
    std::cout << (long double)std::numeric_limits<long double>::max() << std::endl;

    // std::cout << "-- float128_t" << std::endl;
    // std::cout << (long double)std::numeric_limits<float128_t>::min() << std::endl;
    // std::cout << (long double)std::numeric_limits<float128_t>::epsilon() << std::endl;
    // std::cout << (long double)std::numeric_limits<float128_t>::max() << std::endl;

    std::cout << "-- double" << std::endl;
    std::cout << (double)std::numeric_limits<double>::min() << std::endl;
    std::cout << (double)std::numeric_limits<double>::epsilon() << std::endl;
    std::cout << (double)std::numeric_limits<double>::max() << std::endl;

    std::cout << std::numeric_limits<double>::radix << std::endl;
    std::cout << std::numeric_limits<double>::digits << std::endl;
    std::cout << std::numeric_limits<double>::min_exponent << std::endl;
    std::cout << std::numeric_limits<double>::max_exponent << std::endl;

    std::cout << "-- float64_t" << std::endl;
    std::cout << (double)std::numeric_limits<float64_t>::min() << std::endl;
    std::cout << (double)std::numeric_limits<float64_t>::epsilon() << std::endl;
    std::cout << (double)std::numeric_limits<float64_t>::max() << std::endl;

    std::cout << std::numeric_limits<float64_t>::radix << std::endl;
    std::cout << std::numeric_limits<float64_t>::digits << std::endl;
    std::cout << std::numeric_limits<float64_t>::min_exponent << std::endl;
    std::cout << std::numeric_limits<float64_t>::max_exponent << std::endl;

    std::cout << "-- float" << std::endl;
    std::cout << (float)std::numeric_limits<float>::min() << std::endl;
    std::cout << (float)std::numeric_limits<float>::epsilon() << std::endl;
    std::cout << (float)std::numeric_limits<float>::max() << std::endl;

    std::cout << "-- float32_t" << std::endl;
    std::cout << (float)std::numeric_limits<float32_t>::min() << std::endl;
    std::cout << (float)std::numeric_limits<float32_t>::epsilon() << std::endl;
    std::cout << (float)std::numeric_limits<float32_t>::max() << std::endl;

    std::cout << "-- __float16" << std::endl; // unsupported
    std::cout << (float)std::numeric_limits<__float16>::min() << std::endl;
    std::cout << (float)std::numeric_limits<__float16>::epsilon() << std::endl;
    std::cout << (float)std::numeric_limits<__float16>::max() << std::endl;

    std::cout << "-- float16_t" << std::endl;
    std::cout << (float)std::numeric_limits<float16_t>::min() << std::endl;
    std::cout << (float)std::numeric_limits<float16_t>::epsilon() << std::endl;
    std::cout << (float)std::numeric_limits<float16_t>::max() << std::endl;

    std::cout << "-- bfloat16_t" << std::endl;
    std::cout << (float)std::numeric_limits<bfloat16_t>::min() << std::endl;
    std::cout << (float)std::numeric_limits<bfloat16_t>::epsilon() << std::endl;
    std::cout << (float)std::numeric_limits<bfloat16_t>::max() << std::endl;

    std::cout << "-- float8_t" << std::endl;
    std::cout << (float)std::numeric_limits<float8_t>::min() << std::endl;
    std::cout << (float)std::numeric_limits<float8_t>::epsilon() << std::endl;
    std::cout << (float)std::numeric_limits<float8_t>::max() << std::endl;

    std::cout << "-- bfloat8_t" << std::endl;
    std::cout << (float)std::numeric_limits<bfloat8_t>::min() << std::endl;
    std::cout << (float)std::numeric_limits<bfloat8_t>::epsilon() << std::endl;
    std::cout << (float)std::numeric_limits<bfloat8_t>::max() << std::endl;

    return 0;
}
