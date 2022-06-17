#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <tbb/parallel_for_each.h>

int main() {
    std::cout << "Hello Cruel World!" << std::endl;

    std::unordered_map<int, int> m;

    for(int i=0; i<100; ++i)
        m.emplace(i, i);

    tbb::parallel_for_each(m, [](const std::pair<int,int>& p) {
        std::cout << p.first << std::endl;
    });

    std::cout << "Done!" << std::endl;

    return 0;
}
