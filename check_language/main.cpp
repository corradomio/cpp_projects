#include <iostream>

#define self (*this)

struct A {
    int a;

    A& ma() { return self; }
};

struct B : public A {
    int b;

    B& mb() { return self; }
};

int main() {
    std::cout << "Hello, World!" << std::endl;

    B b1, b2;

    b2 = b1.mb();

    return 0;
}
