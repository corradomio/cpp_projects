cmake_minimum_required(VERSION 3.28)
project(check_cudart)


include_directories(
    ./include
    /libs/include
    "D:/Programming/CUDA/v12.2/include"
)

link_directories(
    /libs/lib
    "D:/Programming/CUDA/v12.2/bin"
    "D:/Programming/CUDA/v12.2/lib/x64"
)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_cudart main.cpp)

target_link_libraries(check_cudart
    #    cudart64_12
    #    nvcuda
    cudart
    cuda
)
