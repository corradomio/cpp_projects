cmake_minimum_required(VERSION 3.27)
project(check_linalg)

# C++23 accepts operator[](int, int)
#
set(CMAKE_CXX_STANDARD 23)

add_executable(check_linalg
    main.cpp
    linalg.h
    vec.cpp
    mat.cpp
)
