cmake_minimum_required(VERSION 3.23)
project(check_intrinsics)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=tigerlake")

add_executable(check_intrinsics main.cpp)
