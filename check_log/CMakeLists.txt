cmake_minimum_required(VERSION 3.20)
project(check_log)

set(CMAKE_CXX_STANDARD 14)

include_directories(
    "D:/MinGW/libs/include"
    "D:/Projects.github/cpp_projects/hlsext/include"
)

link_directories(
    "D:/MinGW/libs/lib"
    "D:/Projects.github/cpp_projects/hlsext/lib"
)

add_executable(check_log main.cpp)

target_link_libraries(check_log
    gc
    tbb
    log4cpp
)
