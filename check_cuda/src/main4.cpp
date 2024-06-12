//
// Created by Corrado Mio on 05/06/2024.
//
#include <iostream>
#include <cassert>
#include <stdexcept>
#include <cstdio>
#include <cuda.h>
#include <stdx/tprintf.h>
#include <unistd.h>
#include "cudacpp/cudacpp.h"
#include "cudacpp/cudamem.h"
#include "common.h"

using namespace stdx;
using namespace cudacpp;


int main() {
    tprintf("Hello World\n");
    // sizeof(size_t) = 8
    // sizeof(1LL)    = 8

    cuda_t cu;

    try {
        size_t N = 9LL * 256LL * 1024LL * 1024LL;
        size_t N_THREADS_PER_BLOCK = 512;

        // {
        //     tprintf("allocating\n");
        //     array_t<int8_t> A(14LL*1024LL*1024LL*1024LL, loc_t::device);
        //     tprintf("... sleep\n");
        //     sleep(10);
        //     tprintf("ok\n");
        // }
        // return 0;

        size_t nthreads, nblocks;
        if (N <= N_THREADS_PER_BLOCK) {
            nblocks  = 1;
            nthreads = N;
        } else {
            nblocks  = N / N_THREADS_PER_BLOCK;
            nthreads = N_THREADS_PER_BLOCK;
        }

        assert(N == nblocks*nthreads);

        size_t bytes = N*sizeof(float);

        tprintf("memory allocation: %.02f MB\n", 3.f*float(bytes)/(1024.f*1024.f));
        array_t<float> C(N, loc_t::device);
        array_t<float> A(N, loc_t::host_mapped);
        array_t<float> B(N, loc_t::host_mapped);

        tprintf("array initialization\n");
        // for (int i = 0; i < N; ++i) {
        //     A[i] = 1;
        //     B[i] = 1;
        // }
        std::fill(A.data(), A.data()+A.size(), 1.f);
        std::fill(B.data(), B.data()+B.size(), 1.f);
        float d=0;

        tprintf("    %f\n", sum(A, B, d));

        tprintf("transfer to device\n");
        A.to(loc_t::device);
        B.to(loc_t::device);
        C.to(loc_t::device);

        tprintf("load module\n");
        module_t module("D:/Projects.github/cpp_projects/check_cuda/cu/vecadd.ptx");

        tprintf("call kernel\n");
        module.launch(nblocks, nthreads, "VecAdd", A, B, C, d);

        tprintf("kernel end\n");
        tprintf("transfer to host\n");

        C.to(loc_t::host);
        tprintf("    %f\n", sum(C));

        tprintf("done\n");
    }
    catch (cuda_error& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
