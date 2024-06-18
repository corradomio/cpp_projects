#include <cassert>
#include <iostream>
#include <stdio.h>
#include "stdx/number/i32/numbers.h"
#include "stdx\tprintf.h"

using namespace stdx::number::i32::cns;

int main() {
    std::cout << "Hello, World!" << std::endl;

    int N = 20;
    iset_t M = (iset_t(1) << N);

    stdx::can_tprint(true);
    stdx::tprintf("Start %d ...\n", M);
    for (int S=0; S<M; ++S) {
        int n = ihighbit(S) + 1;
        iset_t L = ilexidx(S, N);
        iset_t T = ilexset(L, N);

        if (stdx::can_tprint()) {
            stdx::tprintf("%3d: %lu -> %lu -> %lu\n", n, S, L, T);
            fflush(stdout);
        }
        assert (S == T);
    }

    stdx::can_tprint(true);
    stdx::tprintf("Done\n");

    return 0;
}
