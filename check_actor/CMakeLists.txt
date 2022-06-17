cmake_minimum_required(VERSION 3.22)
project(check_actor)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    "D:/MinGW/libs/include"
)
link_directories(
    "D:/MinGW/libs/lib"
)

add_executable(check_actor main.cpp)

target_link_libraries(check_actor
    caf_core
    caf_io
)