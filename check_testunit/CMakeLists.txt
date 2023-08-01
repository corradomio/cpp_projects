cmake_minimum_required(VERSION 3.25)
project(check_testunit)

include_directories(
    "D:\\Msys64\\libs\\include"
)

link_directories(
    "D:\\Msys64\\libs\\lib"
)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_testunit main.cpp fact.cpp)

target_link_libraries(check_testunit
    Catch2
    Catch2Main
)
