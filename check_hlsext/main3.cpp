//
// Created by Corrado Mio on 23/09/2020.
//

#include <iostream>
#include <stdx/ranges.h>
#include <stdx/bag.h>

int main3() {

    stdx::bag<int> b;

    for(int i : stdx::range(10))
        b.insert(i);

    for(int i : stdx::range(15))
        b.insert(i);

    for (auto it=b.cbegin(); it != b.cend(); ++it)
        std::cout << it->first << ": " << it->second << std::endl;


    return 0;
}
