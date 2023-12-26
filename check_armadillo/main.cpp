#include <iostream>
#include <armadillo>
#include <openblas/cblas.h>

using namespace std;
using namespace arma;

int main() {
    std::cout << "Hello, World!" << std::endl;

    mat A(4, 5, fill::randu);
    mat B(4, 5, fill::randu);

    cout << A*B.t() << endl;

    return 0;
}
