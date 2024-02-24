cmake_minimum_required(VERSION 3.27)
project(check_cairo)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_cairo main.cpp)

target_link_libraries(check_cairo
    cairo
    opengl32
    gdi32
)
