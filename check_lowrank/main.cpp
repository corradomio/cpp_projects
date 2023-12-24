#include <iostream>
#include "refc.h"
#include "linalg.h"

using namespace linalg;
using namespace stdx;

class BC: public refcount_t { virtual void m() { } };

class D1 : public BC { };
class D2 : public BC { };

int main() {
    std::cout << "Hello, World!" << std::endl;

    tensor m = rand({100, 100});
    tensor v = rand({100});

    tensor M = m;
    tensor V = rand({100});

    return 0;
}
