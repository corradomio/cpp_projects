//
// Created by Corrado Mio on 27/09/2020.
//

#include <iostream>
#include <stdx/params.h>
#include <stdx/ranges.h>


int main8() {

    std::vector<int> p1{1,2,3};
    std::vector<int> p2{4,5,6};

    auto params = stdx::make_params(p1, p2);
    //std::cout << stdx::adder(1);
    //std::cout << stdx::adder(1,2,3,4,5,6,7,8,9);
    for(auto p : params)
        std::cout << std::get<0>(p) << "," << std::get<1>(p) << std::endl;

    return 0;
}