//
// Created by Corrado Mio on 08/06/2024.
//
#include <iostream>
#include <cuda_runtime.h>
#include <stdx/tprintf.h>
#include "common.h"

using namespace stdx;
using namespace cudacpp;


int main51() {
    std::cout << "Hello, World!" << std::endl;

    size_t N = 2ull*1024ull*1024ull*1024ull;
    float *x, *y, *z;
    size_t bytes = N*sizeof(float);

    tprintf("allocate unified memory for %d elements %lld MB\n", N, 3ll*bytes/(1024ll*1024ll));
    // Allocate Unified Memory -- accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    cudaMallocManaged(&z, N*sizeof(float));

    tprintf("initialize x and y arrays\n");
    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 1.0f;
    }
    float c = 1.0f;

    tprintf("load module\n");
    CUmodule hmod;
    check(::cuModuleLoad(&hmod, "D:/Projects.github/cpp_projects/check_cuda/cu/vecadd.ptx"));

    tprintf("retrieve function\n");
    CUfunction hfun;
    check(::cuModuleGetFunction(&hfun, hmod, "VecAdd"));

    tprintf("compute block/grid size\n");
    // Launch kernel on 1M elements on the GPU
    unsigned int numThreads = 1024;
    unsigned int numBlocks = (N + numThreads - 1) / numThreads;
    unsigned int shared_mem = 0;

    dim3  grid_dim{numBlocks};
    dim3 block_dim{numThreads};

    tprintf("launch kernel on %d/%d\n", numBlocks, numThreads);
    void* args[] { &x, &y, &z, &c };
    check(::cuLaunchKernel(
        hfun,
        grid_dim.x, grid_dim.y, grid_dim.z,
        block_dim.x, block_dim.y, block_dim.z,
        shared_mem, 0, args, nullptr
    ));

    tprintf("wait for termination\n");

    tprintf("dispose everything\n");
    check(::cuModuleUnload(hmod));
    ::cudaFree(x);
    ::cudaFree(y);
    ::cudaFree(z);
    return 0;
}
