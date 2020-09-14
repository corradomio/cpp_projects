cmake_minimum_required(VERSION 3.16)
project(check_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)

include_directories(
    deps
)

link_directories(
    deps
)

add_executable(check_cuda
#    main.cu
#    main2.cu
    main3.cu cugraph.h)

set_target_properties(
    check_cuda
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(check_cuda
    igraph
    curand
)