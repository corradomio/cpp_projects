#include <iostream>
#include <arrayfire.h>

int main() {
    std::cout << "Hello, World!" << std::endl;

    af_err err;
    af_array a;
    dim_t dims[1] = {10};

    err = af_create_handle(&a, 1, dims, af_dtype::f32);
    err = af_release_array(a);


    return 0;
}
