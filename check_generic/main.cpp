#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <ref/vector>
#include <ref/unordered_map>
#include <ref/map>
#include <ref/unordered_set>
#include <ref/set>
#include <std/hash>

#include <bohem/vector>

class C {
    static int gid;

    int id;
public:
    C() { id = gid++; printf("C(%d)\n", id); }
    C(int dummy) { id = gid++; printf("C(%d)\n", id); }
    C(const C& c) { id = gid++; printf("C(%d <- %d)\n", id, c.id); }
   ~C() { printf("~C(%d)\n", id); }
};

int C::gid = 0;


void appmain(const std::vector<std::string>& apps) {
    std::cout << "Hello World" << std::endl;

    std::cout
    << std::fixed << double(6045787.14846557) << std::endl
    << std::fixed <<  float(6045787.14846557) << std::endl;

    //std::map<int,C> m;
    //m.emplace(std::pair<int,C>(0, 0));

    //bohem::vector<C> v;
    //for (int i=0; i<10; ++i)
    //    v.emplace_back();
    //
    //bohem::vector<C> w = v;
    //std::cout << "End" << std::endl;

    //typedef std::pair<double, double> coords_t;
    //typedef int uid_t;
    //
    //std::string s = std::to_string(100);

    //ref::vector<C> v;
    //for (int i=0; i<10; ++i)
    //    v.emplace_back();
    //
    //ref::vector<C> w = v;
    //std::hash<double> x;

    //ref::set<coords_t> m;
    //m.emplace(std::make_pair(0,0));
    //m.emplace(std::make_pair(0,0));
    //std::cout << m.size() << std::endl;
    //
    //ref::set<coords_t> p = m;
    //p.emplace(std::make_pair(0,1));
    //std::cout << m.size() << std::endl;


    //std::cout << m[std::make_pair(0,0)] << std::endl;

    //ref::vector<int> v;
    //for (int i=0; i<10; ++i)
    //    v.push_back(i);
    //
    //ref::vector<int> w = v;
    //w[0] += 100;
    //
    //for(int i=0; i<v.size(); ++i)
    //    std::cout << v[i] << std::endl;
}