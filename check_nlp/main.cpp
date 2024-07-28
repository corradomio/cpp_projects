#include <iostream>
#include "text.h"

typedef int int_t, i32_t;

int main() {
    std::cout << "Hello, World!" << std::endl;

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
