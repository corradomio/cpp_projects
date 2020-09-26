//
// Created by Corrado Mio on 26/09/2020.
//

#include <iosfwd>
#include <string>
#include <vector>
#include <set>
#include <iostream>
#include <stdx/to_string.h>

int main6() {

    std::set<std::string> m{"a","b"};
    std::vector<int> v{1,2,3,4,5};
    std::vector<std::set<std::string>> vs{m,m};

    std::string sep = ",";
    std::cout << stdx::str(v, sep) << std::endl;
    std::cout << stdx::str(vs, sep) << std::endl;

    return 0;
}