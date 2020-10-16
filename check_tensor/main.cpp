#include <iostream>
#include <Fastor/Fastor.h>

using namespace Fastor;

enum {I,J,K,L,M,N};

int main() {
    // An example of Einstein summation
    Tensor<double,2,3,5> A; Tensor<double,3,5,2,4> B;
    // fill A and B
    A.random(); B.random();
    auto C = einsum<Index<I,J,K>,Index<J,L,M,N>>(A,B);

    // An example of summing over three indices
    Tensor<double,5,5,5> D; D.random();
    auto E = inner(D);

    // An example of tensor permutation
    Tensor<float,3,4,5,2> F; F.random();
    auto G = permute<Index<J,K,I,L>>(F);

    // Output the results
    print("Our big tensors:",C,E,G);

    return 0;
}