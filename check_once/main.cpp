#include <iostream>
#include <stdx/once.h>

int main() {
    std::cout << "Hello, World!" << std::endl;

    stdx::once<int> value{10};

    value.detach();
    value = 20;

    std::cout  << value << std::endl;

    return 0;
}
