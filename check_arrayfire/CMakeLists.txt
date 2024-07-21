cmake_minimum_required(VERSION 3.30)
project(check_arrayfire)

include_directories(
    "D:/MinGW/libs/arrayfire-3.9.0/include"
)

link_directories(
    "D:/MinGW/libs/arrayfire-3.9.0/lib"
)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_arrayfire main.cpp)

target_link_libraries(check_arrayfire
    af
)
