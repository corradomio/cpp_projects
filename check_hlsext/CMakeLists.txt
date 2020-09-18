cmake_minimum_required(VERSION 3.17)
project(check_hlsext)

set(CMAKE_CXX_STANDARD 14)

include_directories(
    ../hlsext/include
)

link_directories(
    ../hlsext/lib
)

add_executable(check_hlsext main.cpp)

target_link_libraries(check_hlsext
    hlsext
)