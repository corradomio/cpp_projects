cmake_minimum_required(VERSION 3.26)
project(check_ieee754)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_ieee754 main.cpp
    real_t.cpp
    real_t.h
    float_t.h)
