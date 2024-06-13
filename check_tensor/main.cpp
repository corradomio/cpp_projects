#include <iostream>
#include "stdx/tensor.h"

using namespace stdx::linalg;

void init_tensor(tensor_t<float> t, float off=0) {

    switch(t.rank()) {
        case 0:
            t = 1.f;
            break;
        case 1:
            for (uint16 i0 = 0; i0 < t.dim(0); ++i0)
                t[{i0}] = float(i0+off);
            break;
        case 2:
            for (uint16 i0 = 0; i0 < t.dim(0); ++i0)
                for (uint16 i1 = 0; i1 < t.dim(1); ++i1)
                    t[{i0, i1}] = float(i0+off)*100 + float(i1+off);
            break;
        case 3:
            for (uint16 i0 = 0; i0 < t.dim(0); ++i0)
                for (uint16 i1 = 0; i1 < t.dim(1); ++i1)
                    for (uint16 i2 = 0; i2 < t.dim(2); ++i2)
                        t[{i0, i1, i2}] = float(((i0 + off)*100 + (i1 + off))*100 + (i2 + off));
    }
}

void print_tensor(tensor_t<float> t) {
    switch(t.rank()) {
        case 0:
            printf("---- rank: 0\n");
            printf("%f\n ", (float)t);
            break;
        case 1:
            printf("---- rank: 1, dims: (%zu)\n", t.dim(0));
            for (uint16 i0 = 0; i0 < t.dim(0); ++i0) {
                printf("%04.0f ", t[{i0}]);
            }
            break;
        case 2:
            printf("---- rank: 2, dims: (%zu, %zu)\n", t.dim(0), t.dim(1));
            for (uint16 i0 = 0; i0 < t.dim(0); ++i0) {
                for (uint16 i1 = 0; i1 < t.dim(1); ++i1) {
                    printf("%04.0f ", t[{i0, i1}]);
                }
                printf("\n");
            }
            break;
        case 3:
            printf("---- rank: 3, dims: (%zu, %zu, %zu)\n", t.dim(0), t.dim(1), t.dim(2));
            for (uint16 i0 = 0; i0 < t.dim(0); ++i0) {
                for (uint16 i1 = 0; i1 < t.dim(1); ++i1) {
                    for (uint16 i2 = 0; i2 < t.dim(2); ++i2) {
                        printf("%06.0f ", t[{i0, i1, i2}]);
                    }
                    printf("\n");
                }
                printf("----\n");
            }
    }
    printf("\n");
    fflush(stdout);
}


int main() {
    tensor_t<float> b{3,4}, t;
    init_tensor(b);
    print_tensor(b);

    tensor_t<float> v = b.swap(0, 1);
    print_tensor(v);
}


int main15() {

    tensor_t<float> b{10,10,10}, t;
    init_tensor(b);
    b.dump();

    tensor_t<float> v = b.view({{2,4}, {1,3},{3,7}});
    v.dump();
    print_tensor(v);

    tensor_t<float> e = b.view({2,3,4});
    e.dump();
    print_tensor(e);

    return 0;
}


int main14() {

    tensor_t<float> b{3, 4, 5}, t;
    init_tensor(b);
    // print_tensor(b);

    // t = b.view({{1, -1}, {1, -1}});
    // print_tensor(t);
    //
    // t = b.view({{1, 3}, {2, 4}});
    // print_tensor(t);
    //
    // t = b.view({1, 1});
    // print_tensor(t);
    //
    // t = b.view({-1, -1, 3});
    // print_tensor(t);

    // t = b.view({0,0,-1});
    // print_tensor(t);

    t = b.view({2,3,3});
    print_tensor(t);

    return 0;
}



int main13() {

    tensor_t<float> b{9, 9};
    init_tensor(b);
    print_tensor(b);

    tensor_t<float> t = b.view({{1, ANY}, {1, ANY}});
    print_tensor(t);

    t = b.view({{1, 3}, {4, 7}});
    print_tensor(t);

    t = b.view({0, 2});
    print_tensor(t);

    return 0;
}


int main12() {

    tensor_t<float> b{20};
    init_tensor(b);

    // ----------------------------------------------------------------------
    print_tensor(b);

    tensor_t<float> c = b.view({{1, 4}});
    print_tensor(c);

    // ----------------------------------------------------------------------
    tensor_t<float> t = b.view({{0, ANY, 2}});
    print_tensor(t);

    // ----------------------------------------------------------------------
    tensor_t<float> u = t.view({{2, ANY, 2}});
    print_tensor(u);

    return 0;
}


int main11() {

    tensor_t<float> s;
    s = 10;

    printf("rank: %zu, dim[0]: %zu\n", s.rank(), s.dim(0));
    printf("   %f\n", float(s));
    fflush(stdout);

    tensor_t<float> t{5, 3, 4};
    tensor_t<float> u{t};
    tensor_t<float> c{t, true};

    printf("t[%zu,%zu,%zu]\n", t.dim(0), t.dim(1), t.dim(2));
    fflush(stdout);

    init_tensor(t);

    // c = t;
    t = t.view({ANY, ANY, ANY});

    t.dump();
    fflush(stdout);
    return 0;

    printf("----\n");
    for (uint16 i0 = 0; i0 < t.dim(0); ++i0) {
        for (uint16 i1 = 0; i1 < t.dim(1); ++i1) {
            for (uint16 i2 = 0; i2 < t.dim(2); ++i2) {
                printf("%3.0f ", t[{i0, i1, i2}]);
            }
            printf("\n");
        }
        printf("----\n");
    }

    c = t.view({0});

    return 0;
}