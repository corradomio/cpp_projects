#include <iostream>

#include <stdx/ref_map.h>
#include <stdx/ref_vector.h>
#include <stdx/keys.h>

ref::map<int, int> fill_map() {
    ref::map<int, int> map;

    for (int i =0; i<10; ++i) {
        map.insert(i, i*i);
    }

    return map;
}

void dump_map(const ref::map<int, int> map) {
    for (int k : stdx::keys(map.ref()))
        std::cout << k << ":" << map[k] << std::endl;
}



ref::vector<int> fill_vector() {
    ref::vector<int> vect;

    for(int i=0; i<10; ++i)
        vect.insert(i*i);

    return vect;
}

void dump_vector(ref::vector<int> vect) {
    for (int i=0; i<vect.size(); ++i)
        std::cout << i << ":" << vect[i] << std::endl;
}


int main() {
    std::cout << "ref::map" << std::endl;

    ref::map<int, int> map = fill_map();
    dump_map(map);

    std::cout << "ref::vector" << std::endl;

    ref::vector<int> vect = fill_vector();
    dump_vector(vect);

    std::cout << "done" << std::endl;
    return 0;
}
