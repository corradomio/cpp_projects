//
// Created by Corrado Mio on 05/06/2024.
//
#include <iostream>
#include <stdexcept>
#include <cstdio>
#include <cuda.h>
#include "cudacpp/cudacpp.h"
#include "cudacpp/cudamem.h"


using namespace cudacpp;

extern double sum(array_t<float> A, array_t<float> B, float c);
extern double sum(array_t<float> C);
extern void tprintf(const char *__format, ...);



int main21() {
    tprintf("Hello World\n");

    try {
        cudacpp::cuda_t cu;
        size_t N_BLOCKS = 2*1024*1024;
        size_t N_THREADS_PER_BLOCK = 1024;
        size_t N = N_BLOCKS * N_THREADS_PER_BLOCK;
        size_t bytes = 3 * N * sizeof(float);

        tprintf("... required memory: N=%lld, %lld B, %lld MB\n", N, bytes, bytes / (1024 * 1024));
        fflush(stdout);

        size_t nthreads, nblocks;
        if (N <= N_THREADS_PER_BLOCK) {
            nblocks = 1;
            nthreads = N;
        } else {
            nblocks = N / N_THREADS_PER_BLOCK;
            nthreads = N_THREADS_PER_BLOCK;
        }

        tprintf("... memory allocation\n");
        cudacpp::array_t<float> A(N, cudacpp::loc_t::unified);
        cudacpp::array_t<float> B(N, cudacpp::loc_t::unified);

        cudacpp::array_t<float> C(N, cudacpp::loc_t::unified);
        C.to(cudacpp::loc_t::device);

        tprintf("... filling with values\n");
        for (int i = 0; i < N; ++i) {
            // A[i] = float(i);
            // B[i] = float(i);
            A[i] = 1;
            B[i] = 0;
        }
        float c = 0;

        tprintf("... computing the sum (CPU)\n");
        tprintf("... ... sum = %f\n", sum(A, B, c));
        fflush(stdout);

        tprintf("... transfer to device\n");
        A.to(cudacpp::loc_t::device);
        B.to(cudacpp::loc_t::device);

        tprintf("... module loading\n");
        cudacpp::module_t module("D:/Projects.github/cpp_projects/check_cuda/cu/vecadd.ptx");

        tprintf("... ... kernel launch\n");
        module.call(nblocks, nthreads, "VecAdd", A, B, C, c);
        tprintf("... kernel done\n");

        tprintf("... transfer to host\n");
        C.to(loc_t::host);

        tprintf("... computing the sum (CPU)\n");
        tprintf("... ... sum = %f\n", sum(C));
        tprintf("done\n");
        fflush(stdout);
    }
    catch (cuda_error& ce){
        std::cerr << ce.what() << std::endl;
    }

    return 0;
}

// int main21() {
//     tprintf("Hello World\n");
//
//     CUdeviceptr A;
//     CUdeviceptr B;
//     CUdeviceptr C;
//
//     check(::cuInit(0));
//     check(::cuDeviceGet(&dev, 0));
//     check(::cuCtxCreate(&ctx, 0, dev));
//     check(::cuModuleLoad(&hmod, "D:/Projects.github/cpp_projects/check_cuda/cu/vecadd.ptx"));
//     // c++ name mangling
//     // check(::cuModuleGetFunction(&fun, hmod, "_Z6VecAddPfS_S_"));
//     // using 'extern "C" ...'
//     check(::cuModuleGetFunction(&fun, hmod, "VecAdd"));
//     check(::cuMemAllocManaged(&A, 100*sizeof(float), CU_MEM_ATTACH_GLOBAL));
//     check(::cuMemAllocManaged(&B, 100*sizeof(float), CU_MEM_ATTACH_GLOBAL));
//     check(::cuMemAllocManaged(&C, 100*sizeof(float), CU_MEM_ATTACH_GLOBAL));
//
//     for (int i=0; i<100; ++i) {
//         ((float*)A)[i] = 0.f + i;
//         ((float*)B)[i] = 0.f + i;
//     }
//
//     void *args[3] = { &A, &B, &C };
//
//     check(::cuLaunchKernel(fun,
//                      1,1,1,         // grid  x,y,z
//                      100,1,1,       // block x,y,z
//                      0,             // shared mem bytes
//                      0,             // hstream
//                      args,          // kernel params
//                      0));            // extras
//     tprintf("kernel end");
//     for (int i=0; i<10; ++i) {
//         tprintf("... %f\n", ((float*)C)[i]);
//     }
//
//     check(::cuMemFree(A));
//     check(::cuMemFree(B));
//     check(::cuMemFree(C));
//     check(::cuModuleUnload(hmod));
//     check(::cuCtxDestroy(ctx));
//     return 0;
// }