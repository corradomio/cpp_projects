#include <iostream>
#include <stdio.h>

#include "arena.h"

int main() {
    arena_t arena(1024);

    arena_ptr<float> pf = arena.alloc<float>(1);

    *pf = 9.99;

    printf("%.3f\n", *(pf.get()));

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
