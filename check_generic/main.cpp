#include <iostream>
#include <vector>
#include <ref/list>
#include <ref/vector>
#include <ref/set>
#include <ref/unordered_set>
#include <ref/map>
#include <ref/unordered_map>
#include <ref/algos>
//#include <stdx/to_string.h>


void appmain(const std::vector<std::string>& apps) {
    std::cout << "Hello World 1" << std::endl;

    ref::map<int, int> v;
    ref::map<int, int> x = v;
    for (int i=0; i<10; ++i)
        v.put(i, 2*i);
    ref::unordered_map<int, int> w;
    ref::add_all(w, v);

    //ref::set<int> v;
    //ref::set<int> x = v;
    //for (int i=0; i<10; ++i)
    //    v.add(i);
    //ref::unordered_set<int> w;
    //ref::add_all(w, v);

    std::cout << std::to_string(v) << std::endl;
    std::cout << std::to_string(x) << std::endl;
    std::cout << std::to_string(w) << std::endl;

}