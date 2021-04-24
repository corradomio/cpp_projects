cmake_minimum_required(VERSION 3.19)
project(check_json)

include_directories(
    "D:/MinGW/libs/include"
    "D:/Projects.github/cpp_projects/hlsext/include"
)

link_directories(
    "D:/MinGW/libs/lib"
    "D:/Projects.github/cpp_projects/hlsext/lib"
)

set(CMAKE_CXX_STANDARD 14)

add_executable(check_json main.cpp)