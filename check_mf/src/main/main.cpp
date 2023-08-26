#include <iostream>
#include "../library/linalg.h"
#include <armadillo>
#include <openblas/cblas.h>

namespace la = hls::linalg;

int main() {
    std::cout << "Hello, World!" << std::endl;

    la::array<float> v = la::create(100);
    la::array<float> m = la::create(100, 500);
    la::array<float> u;
    la::array<float> p;

    u = v;
    p = m;

    // v[10] = 10;
    // m[10][10] = 100;
    //
    // u = v;
    // p = m;
    //
    // std::cout << v[10] << std::endl;
    // std::cout << m[10][10] << std::endl;
    //
    // std::cout << v(10) << std::endl;
    // std::cout << m(10, 10) << std::endl;

    return 0;
}
