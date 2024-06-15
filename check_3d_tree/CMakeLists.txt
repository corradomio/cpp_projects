cmake_minimum_required(VERSION 3.28)
project(check_3d_tree)

set(CMAKE_CXX_STANDARD 17)


include_directories(
    ./include
    ../commons/include
    /libs/include
)

link_directories(
    /libs/lib
    ../commons/lib
)

add_executable(check_3d_tree
    main.cpp
    octtree.h
)
