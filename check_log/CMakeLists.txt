cmake_minimum_required(VERSION 3.20)
project(check_log)

set(CMAKE_CXX_STANDARD 14)

add_executable(check_log main.cpp)

target_link_libraries(check_log
    gc
    tbb
    spdlog
)
