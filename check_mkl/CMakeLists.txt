cmake_minimum_required(VERSION 3.17)
project(check_mkl)

include_directories(
    "D:/MinGW/libs/include"
    "../hlsext/include"
)

link_directories(
    "D:/MinGW/libs/lib"
)

set(CMAKE_CXX_STANDARD 14)

add_executable(check_mkl main.cpp)

target_link_libraries(check_mkl
    mkl_core.lib
    mkl_intel_thread.lib
    mkl_rt.lib
    mkl_tbb_thread.lib
    mkl_intel_ilp64.lib
    mkl_intel_lp64.lib
)