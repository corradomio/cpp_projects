#include <iostream>

struct C {
    static int gid;
    int id;
    int* data;
    C() {
        id = ++gid;
        data = new int[100];
        std::cout << "constructor" << std::endl;
    }
    C(const C& c){
        id = ++gid;
        data = new int[100];
        std::cout << "copy constructor" << std::endl;
    }
    C(C&& c) {
        id = ++gid;
        data = c.data;
        c.data = nullptr;
        std::cout << "move constructor" << std::endl;
    }
    ~C() {
        delete[] data;
        if (data != nullptr)
            std::cout << "deleted " << id << std::endl;
        else
            std::cout << "deleted empty " << id << std::endl;
    }
};

int C::gid = 0;

C fun(){
    C c;
    for (int i=0; i<100; ++i)
        c.data[i] = i*i;
    return c;
}


int main() {
    std::cout << "Hello, World!" << std::endl;

    C a;
    C b(a);
    C c = b;
    C d = fun();
    C e(fun());
    C f(std::move(a));

    std::cout << e.id << std::endl;

    return 0;
}
