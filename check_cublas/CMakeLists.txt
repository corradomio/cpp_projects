cmake_minimum_required(VERSION 3.27)
project(check_cublas)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    "D:/Programming/CUDA/v12.2/include"
)

link_directories(
    "D:/Programming/CUDA/v12.2/bin"
)

add_executable(check_cublas main.cpp)

target_link_libraries(check_cublas
    cublas64_12
)
