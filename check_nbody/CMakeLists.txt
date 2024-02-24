cmake_minimum_required(VERSION 3.27)
project(check_nbody)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_nbody
        main.cpp
        v3d.h
        v3d.cpp
        array.h
        space.h
)
