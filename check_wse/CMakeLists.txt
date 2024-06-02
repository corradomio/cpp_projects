cmake_minimum_required(VERSION 3.28)
project(check_wse)

include_directories(
    "D:/Projects.github/cpp_projects/commons/include"
)

set(CMAKE_CXX_STANDARD 17)


add_executable(check_wse main.cpp
    include/cerebras.h
    src/cerebras.cpp
    ../commons/include/stdx/intfloat.h
)
