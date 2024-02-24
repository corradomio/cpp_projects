cmake_minimum_required(VERSION 3.27)
project(check_structures)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_structures main.cpp
    array.h
    refcount.h
    refc.h
    octree.h
    octree.cpp
    octree.cpp
    v3d.h
    list.h)
