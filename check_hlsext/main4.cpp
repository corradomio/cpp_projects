//
// Created by Corrado Mio on 24/09/2020.
//

#include <set>
#include <iostream>


int main() {
    std::set<int, std::greater<int>> set{3,5,1,7,4};

    for(int i : set)
        std::cout << i << std::endl;
    return 0;
}