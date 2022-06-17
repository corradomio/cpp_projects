cmake_minimum_required(VERSION 3.17)
project(check_igraph)

set(CMAKE_CXX_STANDARD 14)

include_directories(
    "D:/MinGW/libs/include"
)

link_directories(
    "D:/MinGW/libs/lib"
)

add_executable(check_igraph
    main.cpp
    igraph.cpp igraph.hpp igraph_simple.cpp igraph_dstruct.cpp)

target_link_libraries(check_igraph
    igraph
)