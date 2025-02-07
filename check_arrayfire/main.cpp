#include <iostream>
#include <arrayfire.h>
#include <cstdint>

int main() {
    std::cout << "Hello, World!" << std::endl;
    std::cout << sizeof(af_array) << std::endl;
    std::cout << sizeof(long long) << std::endl;
    std::cout << sizeof(uint64_t) << std::endl;

    // af_array == void* == uint64_t

    af_array a[1];
    dim_t dims[1] = {1};

    af_err err = af_create_handle(a, 1, dims, af_dtype::f32);
    err = af_release_array(a[0]);


    return 0;
}
