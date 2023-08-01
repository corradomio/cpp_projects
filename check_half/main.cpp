#include <iostream>
#include <limits>
#include <cmath>
#include "ieee754.h"
#include "ieee754std.h"
#include "float16_t.h"
#include <math.h>

void test00() {
    ieee754::float32 f(0, 255, 0);
    ieee754::float32 n(0, 255, 1);
    ieee754::float32 m(0, 254, 0x7FFFFF);
    float inf = f;
    float max = m;
    float nan = n;

    std::cout << max << ", " << inf << ", " << nan << std::endl;
    std::cout << "-- + --" << std::endl;
    std::cout << inf+max << std::endl;
    std::cout << max+inf << std::endl;
    std::cout << inf+inf << std::endl;
    std::cout << "--" << std::endl;
    std::cout << max+nan << std::endl;
    std::cout << inf+nan << std::endl;

    std::cout << "-- - --" << std::endl;
    std::cout << inf-max << std::endl;
    std::cout << max-inf << std::endl;
    std::cout << inf-inf << std::endl;
    std::cout << "--" << std::endl;
    std::cout << max-nan << std::endl;
    std::cout << inf-nan << std::endl;
    std::cout << nan-max << std::endl;
    std::cout << nan-inf << std::endl;

    std::cout << "-- * --" << std::endl;
    std::cout << inf*max << std::endl;
    std::cout << max*inf << std::endl;
    std::cout << inf*inf << std::endl;
    std::cout << inf*0. << std::endl;
    std::cout << "--" << std::endl;
    std::cout << inf*nan << std::endl;
    std::cout << max*nan << std::endl;

    std::cout << "-- / --" << std::endl;
    std::cout << inf/max << std::endl;
    std::cout << max/inf << std::endl;
    std::cout << inf/inf << std::endl;
    std::cout << inf/0. << std::endl;
    std::cout << inf/max << std::endl;
    std::cout << 0./inf << std::endl;
    std::cout << "--" << std::endl;
    std::cout << inf/nan << std::endl;
    std::cout << max/nan << std::endl;
    std::cout << nan/inf << std::endl;
    std::cout << nan/max << std::endl;

}

void test01() {

    numeric::float16_t f1(123.);
    numeric::float16_t f2(.456);
    numeric::float16_t f3, f4;

    f3 = f1+f2;
    f4 = f3-f2;

    std::cout << (float)f3 << ", " << (float)f4 << std::endl;
}

void test1s() {

    ieee754::float32 f1, f2, f3;

    f1 = 1;
    f2 = f1+f1;
    std::cout << (float)f2 << ", " << 2 << std::endl;
    //
    // f1 = 123.;
    // f2 = .456;
    // f3 = f1 + f2;
    // std::cout << (float)f3 << ", " << (123.456) << std::endl;
}

void test2s() {
    ieee754::float32 f1(123.);
    ieee754::float32 f2(.456);

    ieee754::float32 f3 = f1 + f2;

    std::cout << (float)f3 << ", " << 123.456 << std::endl;
}

void test3s() {
    ieee754::float32 f1(.456);
    ieee754::float32 f2(123.);

    ieee754::float32 f3 = f1 + f2;

    std::cout << (float)f3 << ", " << 123.456 << std::endl;
}

void test0d() {
    ieee754::float32 f1(1);
    ieee754::float32 f2(.5);

    ieee754::float32 f3 = f1 - f2;

    std::cout << (float)f3 << ", " << .5 << std::endl;
}

void test1d() {
    ieee754::float32 f1(123.456);
    ieee754::float32 f2(.456);

    ieee754::float32 f3 = f1 - f2;

    std::cout << (float)f3 << ", " << 123 << std::endl;
}

void test2d() {
    ieee754::float32 f1(0);
    ieee754::float32 f2(.456);

    ieee754::float32 f3 = f1 - f2;

    std::cout << (float)f3 << ", " << -.456 << std::endl;
}

void test3d() {
    ieee754::float32 f1(-.456);
    ieee754::float32 f2(0);

    ieee754::float32 f3 = f1 - f2;

    std::cout << (float)f3 << ", " << -.456 << std::endl;
}

void test1p() {
    ieee754::float32 f1(0.5);
    ieee754::float32 f2(1.5);

    ieee754::float32 f3 = f1 * f2;

    std::cout << (float)f3 << ", " << (0.5*1.5) << std::endl;
}

void test1q() {
    ieee754::float32 f1(1.5);
    ieee754::float32 f2(2.);

    ieee754::float32 f3 = f1 / f2;

    std::cout << (float)f3 << ", " << (1.5/2.) << std::endl;
}

void test2q() {
    ieee754::float32 f1(1);
    ieee754::float32 f2(2.);

    ieee754::float32 f3 = f1 / f2;

    std::cout << (float)f3 << ", " << (1./2.) << std::endl;
}

void test3q() {
    ieee754::float32 f1(2);
    ieee754::float32 f2(.5);

    ieee754::float32 f3 = f1 / f2;

    std::cout << (float)f3 << ", " << 4 << std::endl;
}

void test4q() {
    ieee754::float32 f(0,255, 0);

    std::cout << ieee754::isfinite( (float)f) <<  std::endl;
    std::cout << ieee754::isinf((float)f) <<  std::endl;
    std::cout << ieee754::isnormal((float)f) <<  std::endl;
    std::cout << ieee754::isnan((float)f) <<  std::endl;
    std::cout << ieee754::signbit((float)f) <<  std::endl;
}

int main11() {
    // printf("%d %d\n", 0^1, 1^1);

    // isnan(0.5);
    //
    // ieee754::float32 inf = ieee754::inf;

    // test00();
    // test01();

    test1s();
    test2s();
    test3s();

    // test0d();
    // test1d();
    // test2d();
    // test3d();

    // test1p();

    test1q();
    test2q();
    test3q();
    test4q();
    return 0;
}