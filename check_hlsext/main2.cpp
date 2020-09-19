//
// Created by Corrado Mio on 19/09/2020.
//
#include <random>
#include <iostream>
#include <stdx/random.h>


int main2() {
    stdx::random_t rnd;

    for (int i=0; i<16; ++i)
        std::cout << rnd.next_int() << "\n";

    for (int i=0; i<16; ++i)
        std::cout << rnd.next_int(100) << "\n";

    for (int i=0; i<16; ++i)
        std::cout << rnd.next_double() << "\n";

    return 0;
}

