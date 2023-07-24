#include <iostream>
#include <limits>
#include <locale>
#include <cstdint>
typedef unsigned char byte;


template<typename T>
inline T at(void* p, size_t pos) {
    T& ref = reinterpret_cast<T&>(((byte*)p)[pos]);
    return ref;
}


int main() {
    int i = 123;
    int j = at<int>(&i, 0);

    std::cout << i << "," << j << std::endl;
    return 0;
}