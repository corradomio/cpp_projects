cmake_minimum_required(VERSION 3.29)
project(check_popcnt)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS = "-msse4.2")
set(CMAKE_C_FLAGS = "-msse4.2")

add_executable(check_popcnt main.cpp)
