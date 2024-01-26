#include <iostream>
#include "real_t.h"

using namespace ieee754;
using namespace std;

struct float8_s {byte s:1; byte e:2; byte m:5; };


int main() {
    std::cout << "Hello, World!"    << std::endl;

    std::cout << sizeof(float64_t ) << std::endl;
    std::cout << sizeof(float32_t ) << std::endl;
    std::cout << sizeof(bfloat16_t) << std::endl;
    std::cout << sizeof(float16_t ) << std::endl;
    std::cout << sizeof(bfloat8_t ) << std::endl;
    std::cout << sizeof(float8_t  ) << std::endl;

    // std::cout << sizeof(byte) << std::endl;
    // std::cout << sizeof(uint8_t) << std::endl;
    // std::cout << sizeof(float8_s) << std::endl;

    return 0;
}
