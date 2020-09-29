#include <iostream>
#include <cereal/cereal.hpp>
#include <gc_cpp.h>
#include <string.h>

int main() {
    //std::cout << "Hello, World!" << std::endl;

    char* p = new (GC) char[100];
    strcpy(p, "Hello World");

    std::cout << p << std::endl;

    return 0;
}
