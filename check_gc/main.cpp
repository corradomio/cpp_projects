#include <iostream>
#include <cereal/cereal.hpp>
#include <bohem/vector>
#include <string.h>

int main() {

    bohem::vector<int> v;

    for(int i=0; i<1000000; ++i)
        v.push_back(i);

    int s = 0;
    for(auto it = v.begin(); it != v.end(); ++it)
        s += *it;

    std::cout << s << std::endl;

    return 0;
}
