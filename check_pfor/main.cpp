#include <iostream>
#include <iomanip>
#include <functional>
#include <string>
#include <unordered_set>
#include <fmt/format.h>


int main(void) {

    int i=10;
    fmt::print("{}\n", std::hash<int>{}(10));


    return 0;
}