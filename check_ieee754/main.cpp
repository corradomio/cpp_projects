#include <iostream>
#include "real_t.h"

using namespace ieee754;
using namespace std;

int main() {
    bfloat16_t f1 = 16;

    return 0;
}


int main1() {
    std::cout << "Hello, World!"    << std::endl;

    std::cout << sizeof(char  ) << std::endl;
    std::cout << sizeof(short ) << std::endl;
    std::cout << sizeof(int   ) << std::endl;
    std::cout << sizeof(long  ) << std::endl;
    std::cout << sizeof(long long ) << std::endl;
    std::cout << sizeof(float ) << std::endl;
    std::cout << sizeof(double) << std::endl;
    std::cout << "--" << std::endl;
    std::cout << sizeof(float64_t ) << std::endl;
    std::cout << sizeof(float32_t ) << std::endl;
    std::cout << sizeof(bfloat16_t) << std::endl;
    std::cout << sizeof(float16_t ) << std::endl;
    std::cout << sizeof(bfloat8_t ) << std::endl;
    std::cout << sizeof(float8_t  ) << std::endl;
    std::cout << sizeof(ieee754_t ) << std::endl;

    // std::cout << sizeof(byte) << std::endl;
    // std::cout << sizeof(uint8_t) << std::endl;
    // std::cout << sizeof(float8_s) << std::endl;

    return 0;
}
