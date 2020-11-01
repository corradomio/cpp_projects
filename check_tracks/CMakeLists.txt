cmake_minimum_required(VERSION 3.17)
project(check_tracks)

include_directories(
    "D:/MinGW/libs/include"
    "../hlsext/include"
    "./include"
)

link_directories(
    "D:/MinGW/libs/lib"
    ../hlsext/lib
    "./lib"
)

set(CMAKE_CXX_STANDARD 14)

add_executable(check_tracks
    ../hlsext/include/stdx/to_string.h
    ../hlsext/include/stdx/format.h
    ../hlsext/include/stdx/default_unordered_map.h
    include/csvstream.h
    src/dworld.h
    src/main.cpp
    src/dworld.cpp
    src/infections.h
    src/infections.cpp
    src/other.cpp src/other.h
    ../../../MinGW/libs/include/cxxopts.h
    src/encounters.cpp
)

target_link_libraries(check_tracks
    boost_filesystem-mt
    boost_serialization-mt
    tbb
    hlsext
)