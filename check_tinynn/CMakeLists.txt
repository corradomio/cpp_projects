cmake_minimum_required(VERSION 3.27)
project(check_tinynn)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    ./include
)

add_executable(check_tinynn main.cpp)
