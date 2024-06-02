#include <iostream>
#include <cuda.h>
#include "cudacpp.h"

using namespace cudacpp;

int main() {
    std::cout << "Hello, World!" << std::endl;

    cuda_device_t cu;
    cuda_capabilities_t cap;

    // printf("count: %d\n", cu.count());
    printf("name: %s\n", cu.name().c_str());
    // printf("compute capability: %g\n", cu.compute_capability());
    // printf("multiprocessors: %d\n", cu.multiprocessors());
    // printf("threads_per_multiprocessor: %d\n", cu.threads_per_multiprocessor());
    // printf("warp_size: %d\n", cu.warp_size());
    // printf("total memory: %lld MB\n", cu.total_memory()/(1024*1024));

    // printf("compute capability: %g\n", cap.compute_capability);
    // printf("multiprocessors: %d\n", cap.multiprocessors);
    // printf("threads_per_multiprocessor: %d\n", cap.max_threads_per_multiprocessor);
    // printf("warp_size: %d\n", cap.warp_size);
    // printf("total memory: %lld MB\n", cap.total_memory_mb);

    for(auto it=ATTRIBUTES.cbegin(); it != ATTRIBUTES.cend(); ++it) {
        std::string name = it->first;
        std::cout << name << ": " << cu.attribute(name) << std::endl;
    }

    return 0;
}
