#include <iostream>

#include <stdx/ref_vector.h>
#include <stdx/ref_map.h>
#include <stdx/ref_set.h>
#include <stdx/ref_unordered_map.h>
#include <stdx/ref_unordered_set.h>
#include <stdx/containers.h>



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



ref::set<int> fill_set() {
    ref::set<int> set;
    for(int i=0; i<10; ++i)
        set.insert(i*i);
    return set;
}
void dump_set(ref::set<int> set) {
    for (auto it = set.begin(); it != set.end(); it++)
        std::cout << (*it) << std::endl;
}



ref::unordered_map<int, int> fill_unordered_map() {
    ref::unordered_map<int, int> map;
    for (int i =0; i<10; ++i) {
        map.insert(i, i*i);
    }
    return map;
}
void dump_unordered_map(const ref::unordered_map<int, int> map) {
    for (int k : stdx::keys(map.ref()))
        std::cout << k << ":" << map[k] << std::endl;
}



ref::unordered_set<int> fill_unordered_set() {
    ref::unordered_set<int> set;
    for(int i=0; i<10; ++i)
        set.insert(i*i);
    return set;
}
void dump_unordered_set(ref::unordered_set<int> set) {
    for (auto it = set.begin(); it !=set.end(); it++)
        std::cout << (*it) << std::endl;
}


void check_containment() {
    {
        std::cout << "-- set --" << std::endl;
        std::set<int> set;
        for(int i=0; i<10; ++i) set.insert(i);
        std::cout << stdx::contains(set, 1) << std::endl;
        std::cout << stdx::contains(set, 10) << std::endl;
    }
    {
        std::cout << "-- unordered_set --" << std::endl;
        std::unordered_set<int> set;
        for(int i=0; i<10; ++i) set.insert(i);
        std::cout << stdx::contains(set, 1) << std::endl;
        std::cout << stdx::contains(set, 10) << std::endl;
    }
    {
        std::cout << "-- map --" << std::endl;
        std::map<int, int> map;
        for(int i=0; i<10; ++i) map[i]= (i*i);
        std::cout << stdx::contains_key(map, 1) << std::endl;
        std::cout << stdx::contains_key(map, 10) << std::endl;
    }
    {
        std::cout << "-- unordered_map --" << std::endl;
        std::unordered_map<int, int> map;
        for(int i=0; i<10; ++i) map[i]= (i*i);
        std::cout << stdx::contains_key(map, 1) << std::endl;
        std::cout << stdx::contains_key(map, 10) << std::endl;
    }
    {

    }
}



int main() {

    // -- ref::vector
    //std::cout << "ref::vector" << std::endl;
    //ref::vector<int> vect = fill_vector();
    //dump_vector(vect);

    // -- ref::map
    //std::cout << "ref::map" << std::endl;
    //ref::map<int, int> map = fill_map();
    //dump_map(map);

    // -- ref::set
    //std::cout << "ref::set" << std::endl;
    //ref::set<int> set = fill_set();
    //dump_set(set);

    // -- ref::unordered_map
    //std::cout << "ref::unordered_map" << std::endl;
    //ref::unordered_map<int, int> unordered_map = fill_unordered_map();
    //dump_unordered_map(unordered_map);

    // -- ref::unordered_set
    //std::cout << "ref::unordered_set" << std::endl;
    //ref::unordered_set<int> unordered_set = fill_unordered_set();
    //dump_unordered_set(unordered_set);

    // check containment
    check_containment();

    std::cout << "done" << std::endl;
    return 0;
}
