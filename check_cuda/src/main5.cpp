//
// Created by Corrado Mio on 08/06/2024.
//
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdx/tprintf.h>

using namespace stdx;


void check(CUresult res) {
    if (res != CUDA_SUCCESS) {
        const int MSG_LEN = 512;
        const char *name = nullptr;
        const char *message = nullptr;
        char stream[MSG_LEN + 2];
        ::cuGetErrorName(res, &name);
        ::cuGetErrorString(res, &message);
        ::snprintf(stream, MSG_LEN, "%s: %s", name, message);
        tprintf("%s", stream);
    }
}



int main() {
    std::cout << "Hello, World!" << std::endl;

    size_t N = 1l*1024*1024;
    float *x, *y, *z;

    tprintf("allocate unified memory for %d elements\n", N);
    // Allocate Unified Memory -- accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    cudaMallocManaged(&z, N*sizeof(float));

    tprintf("initialize x and y arrays on the host\n");
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
    ::cudaDeviceSynchronize();

    tprintf("dispose everything\n");
    check(::cuModuleUnload(hmod));
    ::cudaFree(x);
    ::cudaFree(y);
    return 0;
}
