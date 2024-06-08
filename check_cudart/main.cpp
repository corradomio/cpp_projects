#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


void check(CUresult res) {
    if (res != CUDA_SUCCESS) {
        const int MSG_LEN = 512;
        const char *name = nullptr;
        const char *message = nullptr;
        char stream[MSG_LEN + 2];
        ::cuGetErrorName(res, &name);
        ::cuGetErrorString(res, &message);
        ::snprintf(stream, MSG_LEN, "%s: %s", name, message);
        printf("%s", stream);
    }
}



int main11() {
    std::cout << "Hello, World!" << std::endl;

    size_t N = 1l*1024*1024;
    float *x, *y;

    printf("allocate unified memory\n");
    // Allocate Unified Memory -- accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    printf("initialize x and y arrays on the host\n");
    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 1.0f;
    }
    float c = 1.0f;

    printf("load module\n");
    CUmodule hmod;
    check(::cuModuleLoad(&hmod, "D:/Projects.github/cpp_projects/check_cudart/cu/vecadd.ptx"));

    printf("retrieve function\n");
    CUfunction hfun;
    check(::cuModuleGetFunction(&hfun, hmod, "VecAdd"));

    printf("compute block/grid size\n");
    // Launch kernel on 1M elements on the GPU
    unsigned int blockSize = 256;
    unsigned int numBlocks = (N + blockSize - 1) / blockSize;
    unsigned int shared_mem = 0;

    dim3 grid_dim{numBlocks};
    dim3 block_dim{blockSize};

    printf("launch kernel\n");
    void* args[] { &x, &y, &c };
    check(::cuLaunchKernel(
        hfun,
        grid_dim.x, grid_dim.y, grid_dim.z,
        block_dim.x, block_dim.y, block_dim.z,
        shared_mem, 0, args, nullptr
    ));

    printf("wait for termination\n");
    ::cudaDeviceSynchronize();

    printf("dispose everything\n");
    check(::cuModuleUnload(hmod));
    ::cudaFree(x);
    ::cudaFree(y);
    return 0;
}
