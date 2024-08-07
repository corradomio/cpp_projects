#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <ref/list>
#include <ref/forward_list>
#include <ref/vector>
#include <ref/set>
#include <ref/unordered_set>
#include <ref/map>
#include <ref/unordered_map>
#include <ref/algos>

#include <bohem/allocator>

void test1(void) {

    std::vector<int> vi;
    //std::vector<int> vj;
    for(int i=0; i<10; ++i)
        vi.emplace_back(100+i);
    ref::vector<int> vj;
    ref::copy_all(vi, vj);

    std::cout << std::to_string(vi) << std::endl;
    std::cout << std::to_string(vj) << std::endl;

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

typedef size_t aid_t;
typedef std::pair<long, long> coords_t;
typedef ref::set<aid_t> agents_s;
typedef ref::map<coords_t, agents_s> agents_ms;

inline coords_t make_coords(long lon, long lat) {
    return std::make_pair<>(lon, lat);
}




void appmain1(const std::vector<std::string>& apps) {
    std::cout << "Hello World 1" << std::endl;

    std::default_random_engine generator;
    std::uniform_int_distribution<int> rnd_coords(0,10);

    ::srand(100);

    agents_ms v;
    for (int i=0; i<10000; ++i) {
        long lon = rnd_coords(generator);
        long lat = rnd_coords(generator);
        coords_t c = make_coords(lon, lat);
        v[c].emplace(i % 10);
    }

    std::cout << std::to_string(v) << std::endl;
}