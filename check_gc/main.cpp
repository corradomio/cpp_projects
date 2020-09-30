#include <iostream>
#include <cereal/cereal.hpp>
#include <bohem/allocator.h>
#include <bohem/vector>
#include <bohem/map>
#include <bohem/unordered_map>
#include <bohem/set>
#include <bohem/unordered_set>

int main() {

    bohem::vector<int> v;
    bohem::map<int, int> m;
    bohem::unordered_map<int, int> um;
    bohem::set<int> s;
    bohem::unordered_set<int> us;

    for (int i = 0; i < 1000000; ++i) {
        v.push_back(i);
        m[i] = i*i;
        um[i] = i*i;
        s.insert(i);
        us.insert(i*i);
    }

    int t = 0;
    for(auto it = v.begin(); it != v.end(); ++it)
        t += *it;

    std::cout << t << std::endl;

    return 0;
}
