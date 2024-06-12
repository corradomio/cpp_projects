cmake_minimum_required(VERSION 3.19)
project(check_tbb)

include_directories(
    /libs/include
)

link_directories(
    /libs/lib
)

set(CMAKE_CXX_STANDARD 14)

add_executable(check_tbb main.cpp)

target_link_libraries(check_tbb
    tbb
    gc
)
