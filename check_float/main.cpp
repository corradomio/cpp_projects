#include <iostream>

int main() {
    __float128 f128=128.;
    __float80 f80=80.;
    _Float64 f64=64;
    _Float32 f32=32;
    _Float16 f16=16;
//    _Float8 f8=8;

    std::cout << "Hello, World!" << std::endl;
    std::cout << "float128 " << float(f128) << std::endl;
    std::cout << "float80 " << float(f80) << std::endl;
    std::cout << "float64 " << float(f64) << std::endl;
    std::cout << "float32 " << float(f32) << std::endl;
    std::cout << "float16 " << float(f16) << std::endl;
    return 0;
}
