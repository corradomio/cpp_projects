//
// Created by Corrado Mio on 14/06/2024.
//
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <cstdio>
#include <stdx/tprintf.h>
#include "common.h"

using namespace stdx;
using namespace cudacpp;


int main() {

    cuda_t cu;

    try {
        size_t N = 1ull * 256ull * 1024ull * 1024ull / 1024ull;
        size_t N_THREADS_PER_BLOCK = 512;

        size_t nthreads, nblocks;
        if (N <= N_THREADS_PER_BLOCK) {
            nblocks = 1;
            nthreads = N;
        } else {
            nblocks = N / N_THREADS_PER_BLOCK;
            nthreads = N_THREADS_PER_BLOCK;
        }

        dim_t gridDim(nblocks);
        dim_t blockDim(nthreads);

        assert(gridDim.size() == nblocks);
        assert(blockDim.size() == nthreads);
        assert(N == nblocks * nthreads);

        size_t bytes = N * sizeof(float);
        tprintf("memory allocation: %.02f MB\n", 3.f * float(bytes) / (1024.f * 1024.f));

        cudacpp::array_t<float> A(N, loc_t::host);
        cudacpp::array_t<float> B(N, loc_t::host);
        cudacpp::array_t<float> C(N, loc_t::host);

        tprintf("array initialization\n");
        std::fill(A.data(), A.data() + A.size(), 1.f);
        std::fill(B.data(), B.data() + B.size(), 1.f);
        float d = 0;

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
        return 0;
    }
    catch (cuda_error &e) {
        std::cerr << e.what() << std::endl;
    }
    tprintf("end\n");
    fflush(stdout);

    return 0;
}