cmake_minimum_required(VERSION 3.28)
project(check_cuda)


include_directories(
    /libs/include
    "D:/Programming/CUDA/v12.4/include"
)

link_directories(
    /libs/lib
    "D:/Programming/CUDA/v12.4/bin"
)


set(CMAKE_CXX_STANDARD 17)

add_executable(check_cuda main.cpp
    cudacpp.h
    cudacpp.cpp)


target_link_libraries(check_cuda
#    cudart64_12
    nvcuda
)