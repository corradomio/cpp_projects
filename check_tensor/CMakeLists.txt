cmake_minimum_required(VERSION 3.17)
project(check_tensor)

set(CMAKE_CXX_STANDARD 14)

include_directories(
    "D:/MinGW/libs/include"
    "../hlsext/include"
    "./include"
)

link_directories(
    "D:/MinGW/libs/lib"
    #    ../hlsext/lib
    "./lib"
)

add_executable(check_tensor main.cpp)