cmake_minimum_required(VERSION 3.25)
project(check_assignment)

include_directories(
    "D:/Projects.github/cpp_projects/hlsext/include"
)

link_directories(
        "D:/Projects.github/cpp_projects/hlsext/lib"
)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_assignment main.cpp ../hlsext/include/stdx/once.h)

target_link_libraries(check_assignment
    hlsext
)

