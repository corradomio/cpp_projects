#include <iostream>
#include <vector>
#include "text.h"
#include <stdx/containers.h>
#include <stdx/ranges.h>


int main() {
    std::cout << "Hello, World!" << std::endl;

    std::vector<int> v;
    std::set<int> s;
    std::unordered_set<int>  u;

    for(int i : stdx::range(0, 10, 2)) {
        std::cout << i << std::endl;
        stdx::add(v, i);
        stdx::add(s, i);
        stdx::add(u, i);
    }

    text_t t(R"(D:\Projects.github\cpp_projects\check_nlp\La Sacra Bibbia.txt)");
    // t.parse("(\\w+|[.,;:!?])");
    t.parse("(\\w+)");

    std::cout << "text length: " << t.length() << std::endl;
    std::cout << "# terms : " << t.terms().size() << std::endl;
    std::cout << "# tokens: " << t.tokens().size() << std::endl;

    // int i=0;
    // for(const auto term : t.terms()) {
    //     std::cout << term << std:: endl;
    //     i += 1;
    //     if (i >= 100) break;
    // }

    for(const auto token : t.tokens()) {
        if (token.second >= 15000)
        std::cout << token.first << ": " << token.second << std:: endl;
    }

    return 0;
}
