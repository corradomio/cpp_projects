#include <iostream>
#include "real_t.h"

using namespace ieee754;
using namespace std;


int main() {
    std::cout << "Hello, World!"    << std::endl;

    std::cout << sizeof(float64_t ) << std::endl;
    std::cout << sizeof(float32_t ) << std::endl;
    std::cout << sizeof(bfloat16_t) << std::endl;
    std::cout << sizeof(float16_t ) << std::endl;
    std::cout << sizeof(bfloat8_t ) << std::endl;
    std::cout << sizeof(float8_t  ) << std::endl;

    return 0;
}
