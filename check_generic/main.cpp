#include <iostream>
//#include <vector>
#include <algorithm>
#include <ref/list>
#include <ref/forward_list>
#include <ref/vector>
#include <ref/set>
#include <ref/unordered_set>
#include <ref/map>
#include <ref/unordered_map>
#include <ref/algos>
//#include <stdx/to_string.h>


void appmain(const std::vector<std::string>& apps) {
    std::cout << "Hello World 1" << std::endl;

    //std::vector<int> vi;
    //std::vector<int> vj;
    //for(int i=0; i<10; ++i)
    //    vi.emplace_back(100+i);
    //std::cout << std::to_string(vi) << std::endl;
    //std::cout << std::to_string(vj) << std::endl;

    //ref::map<int, int> v;
    //ref::map<int, int> x = v;
    //for (int i=0; i<10; ++i)
    //    v.put(i, 2*i);
    //ref::unordered_map<int, int> w;
    //ref::add_all(w, v);

    ref::list<int> v;
    ref::list<int> x = v;
    for (int i=0; i<10; ++i)
        v.emplace(i);
    ref::list<int> w;
    ref::copy_all(v, w);

    std::cout << std::to_string(v) << std::endl;
    std::cout << std::to_string(x) << std::endl;
    std::cout << std::to_string(w) << std::endl;

}