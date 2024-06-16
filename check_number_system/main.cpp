#include <cassert>
#include <iostream>
#include <stdio.h>
#include "stdx\numbers.h"
#include "stdx\tprintf.h"

using namespace stdx::number::cns;

int main() {
    std::cout << "Hello, World!" << std::endl;

    int N = 20;
    long M = (1 << N);

    stdx::tprintf("Start %d ...\n", M);
    for (int S=0; S<M; ++S) {
        int n = ihighbit(S) + 1;
        int L = stdx::number::cns::ilexidx(S, N);
        int T = stdx::number::cns::ilexset(L, N);
        // printf("%3d: %4d -> %4d -> %4d\n", n, S, L, T);
        // fflush(stdout);
        assert (S == T);
    }
    stdx::tprintf("Done\n");

    return 0;
}
