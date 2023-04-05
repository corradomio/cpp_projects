cmake_minimum_required(VERSION 3.25)
project(check_parallel_kernel)

set(CMAKE_CXX_STANDARD 17)


include_directories(
    "D:/MinGW/libs/include"
)

link_directories(
    "D:/MinGW/libs/lib"
)


add_executable(check_parallel_kernel main.cpp kernel.h)

target_link_libraries(check_parallel_kernel
    tbb12
    pthread
    async++
)

