#include <cstdio>
#include <iostream>

#define self (*this)

template<typename T>
typename T::super& up_cast(T& object) {
    return reinterpret_cast<typename T::super&>(object);
}

struct A {
    int a;

    A() {
        // printf("A()\n");
    }

    ~A() {
        // printf("~A()\n");
    }

    A& ma() {
        printf("ma()\n");
        return self;
    }

    void call() {
        printf("A::call()\n");
    }
};

struct B : public A {
    using super = A;

    int b;

    B() {
        // printf("B()\n");
    }

    ~B() {
        // printf("~B()\n");
    }

    void call(int i) {
        printf("B::call(%d)\n", i);
    }

    B& mb() {
        printf("ma()\n");
        return self;
    }
};

int main() {
    std::cout << "Hello, World!" << std::endl;

    // static_cast
    // const_cast
    // reinterpret_cast
    // dynamic_cast

    B b1;

    b1.ma();
    b1.mb();

    // static_cast<A>(b1).call();
    reinterpret_cast<A&>(b1).call();
    up_cast(b1).call();
    b1.call(1);

    return 0;
}
