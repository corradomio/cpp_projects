#include <iostream>
// #include "half.hpp"
// #include <stdint.h>
// #include <float.h>
// #include <stdfloat>
// #include <cstdint>
// #include <cfloat>
// #include "bfloat16_t.h"

#include "floatx.h"

int main(){

    numeric::float32_t f1 = 12.5;
    numeric::float32_t f2 = f1;
    numeric::float32_t f3 = f1 + 0.33f;
    numeric::float32_t f4 = 0.33f + f1;
    numeric::float32_t f5 = f2 + f3 + f4;

    long i6 = f5.to_bits();
    numeric::float32_t f6 = numeric::float32_t::from_bits(i6);

    std::cout << f1 << std::endl;
    std::cout << f2 << std::endl;
    std::cout << f3 << std::endl;
    std::cout << f4 << std::endl;
    std::cout << f5 << std::endl;

    std::cout << std::hex << i6 << std::endl;
    std::cout << f6 << std::endl;

    return 0;
}
