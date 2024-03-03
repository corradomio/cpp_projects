//
// Created by Corrado Mio on 25/02/2024.
//
#include "../array.h"
#include "tensor.h"

using namespace stdx;

int main2() {

    tensor_t<float> t1;
    tensor_t<float> t2{2,3,5};
    tensor_t<float> t3 = t2.clone();
    tensor_t<float> t4(t2);
    tensor_t<float> t5(t2, true);
    tensor_t<float> t6 = t5.reshape({dims_t::NO_DIM, 3, 2});
    tensor_t<float> t0{0};

    t0.dump();
    t1.dump();  // 1,1
    t2.dump();  // 1,1
    t3.dump();  // 2,1
    t4.dump();  // 3,2
    t5.dump();
    t6.dump();

    return 0;
}