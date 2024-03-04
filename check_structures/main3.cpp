//
// Created by Corrado Mio on 26/02/2024.
//
#include <iostream>
#include <stdio.h>
// #include "array.h"
#include "vector.h"
#include "matrix.h"
#include "vector_op.h"
#include "matrix_op.h"
#include "dot_op.h"


using namespace stdx;

int main() {

    vector_t<float> v1 = ones<float>(10);
    vector_t<float> v2 = ones<float>(10);
    matrix_t<float> m1 = ones<float>(10, 5);
    matrix_t<float> m2 = ones<float>(5, 10);
    matrix_t<float> m3;

    std::cout << dot(v1, v2) << std::endl;
    std::cout << v1.dot(v2) << std::endl;

    v2 = m1.dot(v1);
    v2 = v1.dot(m1);
    m3 = m1.dot(m2);

}


int main2() {
    printf("Hello Cruel World\n");

    vector_t<float> v(3);
    vector_t<float> o = ones<float>(3);

    v += 1.f;
    v -= 1.f;
    v *= 1.f;
    v /= 1.f;

    v += o;
    v -= o;
    v *= o;
    v /= o;

    v = v + o;
    v = v - o;
    v = v * o;
    v = v / o;

    v = v + 1.f;
    v = v - 1.f;
    v = v * 1.f;
    v = v / 1.f;

    v = 1.f + v;
    v = 1.f - v;
    v = 1.f * v;
    v = 1.f / v;

    v = v + v;
    v = v - v;
    v = v * v;
    v = v / v;

    for(int i=0; i<v.size(); ++i)
        printf("  %.3f\n", v[i]);

    matrix_t<float> m(5, 3);
    matrix_t<float> q = ones<float>(5, 3);

    m += 1.f;
    m -= 1.f;
    m *= 1.f;
    m /= 1.f;

    m += q;
    m -= q;
    m *= q;
    m /= q;

    m = m + q;
    m = m - q;
    m = m * q;
    m = m / q;

    m = m + 1.f;
    m = m - 1.f;
    m = m * 1.f;
    m = m / 1.f;

    m = 1.f + m;
    m = 1.f - m;
    m = 1.f * m;
    m = 1.f / m;

    m = m + m;
    m = m - m;
    m = m * m;
    m = m / m;


    return 0;
}

int main1() {

    array_t<float> a(100);

    array_t<float> b;

    b = a;

    a.size(8);
    for(int i=0; i<8; ++i)
        a.at(-(i+1)) = i;

    b[0] = 100;
    b.at(-1) = 99;

    for(int i=0; i<a.size(); ++i) {
        printf("%f\n", a[i]);
        fflush(stdout);
    }

    return 0;
}