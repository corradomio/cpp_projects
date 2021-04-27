#include <iostream>
#include <vector>
#include <algorithm>
#include <future>
#include "parallel.h"

int main() {
    std::cout << "Hello, World!" << std::endl;

    std::vector<int> v;
    stdx::parallel_for(0, 10, [&](auto i){
        std::cout << "c:" << i << std::endl;
        v.emplace_back(i);
    });

    stdx::parallel_foreach(v, [&](auto i) {
        std::cout << "v:" << i << std::endl;
    });

    stdx::parallel_foreach(v.cbegin(), v.cend(), [&](auto i) {
        std::cout << "i:" << i << std::endl;
    });

    std::for_each(
        std::execution::par_unseq, v.begin(), v.end(), [](auto&& item) {
            //do stuff with item
        });

}
