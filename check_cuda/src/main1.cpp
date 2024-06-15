#include <iostream>
#include <cuda.h>
#include "cudacpp/cudacpp.h"
#include "cudacpp/cudamem.h"

using namespace cudacpp;

int main12() {
    cuda_t cu;

    std::cout << "Hello, World!" << std::endl;
    std::cout << sizeof(void*) << ", " << sizeof(long long) << std::endl;

    array_t<int> h(100, loc_t::host);
    array_t<int> d(100, loc_t::device);

    try {
        module_t m("D:\\Projects.github\\cpp_projects\\check_cuda\\cu\\vecadd.ptx");
    }
    catch (cuda_error& e) {
        std::cerr << e.what() << std::endl;
    }
    catch (std::bad_alloc& ba) {

    }


    // buffer_t b;
    // b = mem_alloc(128, host);
    // b = mem_alloc(128, device);
    // b = mem_alloc(128, page_locked);
    // b = mem_alloc(128, unified);

    return 0;
}

int main11() {
    std::cout << "Hello, World!" << std::endl;

    cuda_t cu;
    cuda_attributes_t cap;

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
