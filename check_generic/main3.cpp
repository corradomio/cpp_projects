//
// Created by Corrado Mio (Local) on 01/07/2021.
//
#include <iostream>
#include <vector>
#include <string>

struct Base {
    int fi;

    Base(int i):fi(i){ }
};

struct Derived : public Base {
    int fj;

    Derived(int i, int j):Base(i),fj(j){ }
};

void appmain2(const std::vector<std::string>& apps){
    Base b{1};
    Derived d{2,3};
    Base c{44};

    c = d;
    c = Base(33);
    c = (Base&)d;
    c = Base(11);
    c = *(Base*)(&d);
    c = Base(22);
    c = (Base)d;

    std::cout << c.fi << std::endl;
}
