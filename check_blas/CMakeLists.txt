cmake_minimum_required(VERSION 3.27)
project(check_blas)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    "D:/Programming/CUDA/v12.2/include"
)

add_executable(check_blas main.cpp)
