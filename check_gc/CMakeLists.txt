cmake_minimum_required(VERSION 3.17)
project(check_gc)

set(CMAKE_CXX_STANDARD 14)

include_directories(
    /userlibs/include
)

link_directories(
    /userslibs/lib
)

add_executable(check_gc main.cpp)