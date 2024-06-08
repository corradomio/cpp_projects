//
// Created by Corrado Mio on 05/06/2024.
//
#include <iostream>
#include <stdexcept>
#include <cstdio>
#include <cuda.h>
#include "cudacpp.h"
#include "cudamem.h"

using namespace cudacpp;

extern double sum(array_t<float> A, array_t<float> B, float c);
extern double sum(array_t<float> C);


int main41() {
    printf("Hello World\n\n");

    cuda_t cu;

    try {
        size_t N = 16 * 1024;
        size_t N_THREADS_PER_BLOCK = 1024;

        size_t nthreads, nblocks;
        if (N <= N_THREADS_PER_BLOCK) {
            nblocks  = 1;
            nthreads = N;
        } else {
            nblocks  = N / N_THREADS_PER_BLOCK;
            nthreads = N_THREADS_PER_BLOCK;
        }

        // array_t<float> A(100, loc_t::host);
        array_t<float> A(N, loc_t::unified);
        array_t<float> B(N, loc_t::unified);
        array_t<float> C(N, loc_t::device);

        for (int i = 0; i < N; ++i) {
            A[i] = float(i);
            B[i] = float(i);
        }

        printf("%f\n", sum(A, B, 3.f));
        fflush(stdout);

        A.to(loc_t::device);
        B.to(loc_t::device);

        module_t module("D:/Projects.github/cpp_projects/check_cuda/cu/vecadd.ptx");

        module.call(nblocks, nthreads, "VecAdd", A, B, C, 3.f);

        printf("\nkernel end\n\n");

        C.to(loc_t::host);
        printf("%f\n\n", sum(C));
        fflush(stdout);
    }
    catch (cuda_error& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
