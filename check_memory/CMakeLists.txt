cmake_minimum_required(VERSION 3.20)
project(check_memory)

include_directories(
    "D:/MinGW/libs/include"
    "D:/Projects.github/cpp_projects/hlsext/include"
)

link_directories(
    "D:/MinGW/libs/lib"
    "D:/Projects.github/cpp_projects/hlsext/lib"
)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_memory main.cpp)

target_link_libraries(check_memory
    gc
)