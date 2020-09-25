//
// Created by Corrado Mio on 24/09/2020.
//

#include <set>
#include <iostream>
#include <stdx/ranges.h>
#include <tuple>
#include <stdx/default_unordered_map.h>

int main() {
    std::set<int, std::greater<int>> s1{3,5,1,7,4};
    std::set<int, std::less<int>> s2{3,5,1,7,4};

    stdx::default_unordered_map<int, int> map(3);

    map[1] = 11;
    int x = map.at(5);

    std::cout << map.at(1) << ", " << map.at(3) << std::endl;

    for(auto it = map.cbegin(); it != map.cend(); it++)
        std::cout << it->first << ": " << it->second << std::endl;

    return 0;
}