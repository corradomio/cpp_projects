cmake_minimum_required(VERSION 3.26)
project(check_triton_emu)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_triton_emu main.cpp
    triton_emu.h)
