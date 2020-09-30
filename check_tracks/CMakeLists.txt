cmake_minimum_required(VERSION 3.17)
project(check_tracks)

include_directories(
    "D:/MinGW/userlibs/include"
    "../hlsext/include"
    "./include"
)

link_directories(
    "D:/MinGW/userlibs/lib"
    #    ../hlsext/lib
    "./lib"
)

set(CMAKE_CXX_STANDARD 14)

add_executable(check_tracks
    include/csvstream.h
    src/dworld.h
    src/main.cpp
    src/dworld.cpp
    ../hlsext/include/stdx/to_string.h ../hlsext/include/stdx/format.h src/infections.h src/infections.cpp src/other.cpp src/other.h ../hlsext/include/stdx/default_unordered_map.h)

target_link_libraries(check_tracks
    gc
    gccpp
    boost_filesystem-mt
    boost_serialization-mt
    tbb
#    hlsext
)