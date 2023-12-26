cmake_minimum_required(VERSION 3.17)
project(check_mkl)

include_directories(
    "D:\\MinGW\\libs\\include"
    "D:\\MinGW\\libs\\mkl\\include"
)

link_directories(
    "D:\\MinGW\\libs\\lib"
    "D:\\MinGW\\libs\\mkl\\lib"
)

set(CMAKE_CXX_STANDARD 14)

add_executable(check_mkl main.cpp
    main.h)

target_link_libraries(check_mkl
    mkl_core
    mkl_intel_thread
    mkl_rt
    mkl_tbb_thread
    mkl_intel_ilp64
    mkl_intel_lp64
)