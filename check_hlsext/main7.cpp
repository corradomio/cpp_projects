//
// Created by Corrado Mio on 26/09/2020.
//
#include <iostream>
#include <stdx/ranges.h>

int main() {

    for(int i : stdx::range(1,5))
        std::cout << i << std::endl;


    return 0;
}
