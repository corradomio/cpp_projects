#include <iostream>

#include "triton_emu.h"

using namespace hls::triton;

int main() {
    std::cout << "Hello, World!" << std::endl;

    hls::triton::algorithm_t algo(dim(), dim(16,16), dim(4,4));

    algo.run();

    return 0;
}
