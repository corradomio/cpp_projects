cmake_minimum_required(VERSION 3.25)
project(check_half)

set(CMAKE_CXX_STANDARD 20)

add_executable(check_half main.cpp float16_t.h minifloat.h bfloat16_t.h float24.h floatx.h)
