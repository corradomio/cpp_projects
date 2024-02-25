cmake_minimum_required(VERSION 3.27)
project(check_structures)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_structures main1.cpp
    _suspended/refcount.h
    refc.h
    octree.h
    octree.cpp
    octree.cpp
    _suspended/v3d.h
    list.h
    array.h
    main2.cpp
    tensor.h)
