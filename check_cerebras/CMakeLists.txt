cmake_minimum_required(VERSION 3.27)
project(check_cerebras)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_cerebras main.cpp
    cerebras.h)
