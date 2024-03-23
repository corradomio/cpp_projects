//
// Created by Corrado Mio on 22/03/2024.
//
#include <iostream>
#include "cuda/cublas.h"
#include "cuda/dot_op.h"
using namespace cuda;

int main() {
    matrix_t m1 = range(5,4);
    matrix_t m2 = range(4,5);

    m1.to(GPU);
    m2.to(GPU);

    matrix_t r = dot(m1, m2);

    r.to(CPU);
    print(r);
}


int main33() {
    context.create();

    matrix_t m = range(5,4);
    // vector_t u = range(4);
    vector_t u = range(5);

    print(m);
    print(u);

    m.to(device_t::GPU);
    u.to(device_t::GPU);

    // vector_t r = dot(m, u);
    vector_t r = dot(u, m);

    r.to(device_t::CPU);
    print(r);

    return 0;
}


int main32() {
    context.create();

    matrix_t m;
    matrix_t m1 = range(100, 200);
    matrix_t m2{100, 200, GPU};

    m1 = range(100, 200);

    m1.to(GPU);
    m1.to(CPU);

    for(size_t i=0; i<5; ++i)
        std::cout << m1[i,0] << std::endl;

    return 0;
}


int main31() {

    vector_t u{};
    vector_t v1  = range(100);
    vector_t v2{100, GPU};

    v1 = range(100);

    v1.to(GPU);
    v1.to(CPU);

    for(size_t i=0; i<5; ++i)
        std::cout << v1[i] << std::endl;

    return 0;
}