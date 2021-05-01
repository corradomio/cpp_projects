#include <iostream>
#include <iomanip>
#include <functional>
#include <string>
#include <unordered_set>
#include <fmt/format.h>
#include <stdx/hash.h>
#include <boost/container_hash/hash.hpp>


int main(void) {

    std::hash<float>{}(1.f);
    std::pair<int, int> p{1,2};
    fmt::print("{}\n", std::hash<std::pair<int, int>>{}(p));


    return 0;
}