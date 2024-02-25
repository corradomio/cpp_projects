#include <iostream>
#include <random>
// #include "octree.h"
#include "_suspended/array.h"
#include "list.h"

using namespace stdx;
// using namespace stdx::octree;


struct B {
    int m(int i) { return i+1; }
    // virtual ~B(){}
};

struct D : public B {
    int m(int i) { return B::m(i); }
    int m(int i, int j) { return B::m(i+j+1); }
};


int main1() {

    D d;

    printf("%d\n", d.m(1));
    printf("%d\n", d.m(1,2));


    // std::cout << "Hello, World!" << std::endl;
    //
    // array_t<int> a;
    //
    // for (int i=0; i<100; ++i)
    //     a.add(i);
    //
    // list_t<int> l;
    //
    // for (int i=0; i<100; ++i)
    //     l.add(i);
    //
    //
    // array_t<int> b(10, 0);
    // for (int i=0;i<b.size(); ++i)
    //     std::cout << b[i] << std::endl;


    // octree_t tree(1., 128);
    //
    // for(int i=0; i<100000; ++i) {
    //     element_t e;
    //     e.p.x = float(::rand())/RAND_MAX;
    //     e.p.y = float(::rand())/RAND_MAX;
    //     e.p.z = float(::rand())/RAND_MAX;
    //     e.m   = float(::rand())/RAND_MAX;
    //
    //     tree.add(e);
    // }
    // // tree.dump();
    //
    // tree.save("data.csv");

    return 0;
}
