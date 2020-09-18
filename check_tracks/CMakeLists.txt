cmake_minimum_required(VERSION 3.17)
project(check_tracks)

include_directories(
    "./include"
    "D:/MinGW/userlibs/include"
    ../hlsext/include
)

link_directories(
    "./lib"
    "D:/MinGW/userlibs/lib"
#    ../hlsext/lib
)

set(CMAKE_CXX_STANDARD 14)

add_executable(check_tracks
    include/csvstream.h
    include/dworld.h
    src/main.cpp
    src/dworld.cpp
)

target_link_libraries(check_tracks
#    gcmt-lib
    boost_filesystem-mt
    boost_serialization-mt
    tbb
#    hlsext
)