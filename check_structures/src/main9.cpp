//
// Created by Corrado Mio on 05/06/2024.
//
#include <chrono>
#include <cstdio>
#include <iostream>
#include "stdx/array.h"

int main() {

    stdx::array_t<int> a(10);
    stdx::array_t<int> b(20);
    stdx::array_t<int> c(b);
    stdx::array_t<int> d(c, true);
    a = b;
    a = b.clone();

    return 0;
}
