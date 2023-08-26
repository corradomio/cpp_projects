cmake_minimum_required(VERSION 3.26)
project(check_mf)

set(CMAKE_CXX_STANDARD 23)

add_executable(check_mf
    src/main/main.cpp
    src/library/linalg.h
    src/library/linalg.cpp
)
