cmake_minimum_required(VERSION 3.25)
project(check_asm)

set(CMAKE_CXX_STANDARD 23)

add_compile_options(-masm=intel)

add_executable(check_asm main.cpp)
