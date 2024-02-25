#include <cstdio>
#include "linalg/linalg.h"

using namespace std;
using namespace stdx::linalg;


int main() {
    vector v = ones(3);
    matrix m = stdx::linalg::identity(3);
    vector r = m.dot(v);
    // matrix a = m.dot(2*ones(3,3));

    // float x = v.at(2);
    // float y = m.at(1);
    // float z = m.at(0, 1);

    // v.print();
    // m.print();
    // r.print();
    // a.print();

    matrix m23 = range(2, 3);
    matrix m32 = m23.reshape(3, 2);

    // matrix d = c.transpose();

    // b.print();
    // c.print();
    // d.print();

    vector v2(2);
    vector v3(3);

    r = m23.dot(v3).print();
    r = v2.dot(m23).print();

    return 0;
}

