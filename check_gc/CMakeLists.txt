cmake_minimum_required(VERSION 3.17)
project(check_gc)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    "D:/MinGW/userlibs/include"
    "../hlsext/include"
)

link_directories(
    "D:/MinGW/userlibs/lib"
    "../hlsext/lib"
)

add_executable(check_gc main.cpp)

target_link_libraries(check_gc
    gc-lib
)