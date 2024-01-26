#include <iostream>
#include <stdio.h>
#include "linalg.h"

using namespace std;
using namespace stdx;
using namespace stdx::linalg;


int main() {
    std::cout << "Hello, World!" << std::endl;
    std::cout << "sizeof(int)    " << sizeof(int)    << std::endl;
    std::cout << "sizeof(size_t) " << sizeof(size_t) << std::endl;

    // std::cout << INT32_MAX << std::endl;
    // std::cout << (INT32_MAX+1) << std::endl;

    tensor<float> s;
    tensor<float> v{10};
    tensor<float> m{10, 20};
    tensor<float> t{30,20,10};
    tensor<float> u{5,4,30,20,10};

    tensor<float> r0 = v[{3}];
    tensor<float> r1 = v[{all, 3}];
    tensor<float> r2 = v[{3, all}];

    return 0;
}




//struct A : public stdx::refc_t {
//    virtual void say() {
//        std::cout << "I am A\n";
//    }
//};
//struct B : public A {
//    void say() override {
//        std::cout << "I am B\n";
//    }
//};
//
//int main1() {
//    std::cout << "Hello, World!" << std::endl;
//
//    ref_ptr<A> pa = new B();
//    ref_ptr<B> pb;
//
//    pb = ref_cast<B>(pa);
//    pa = ref_cast<A>(pb);
//
//    pa->say();
//
//    return 0;
//}
