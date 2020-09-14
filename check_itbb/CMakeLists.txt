# cmake_minimum_required(VERSION <specify CMake version here>)
project(check_itbb)

set(CMAKE_CXX_STANDARD 14)

add_executable(check_itbb main.cpp)

target_link_libraries(check_itbb
    tbb
)