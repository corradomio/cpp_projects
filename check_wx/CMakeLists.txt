cmake_minimum_required(VERSION 3.21)
project(check_wx)

set(CMAKE_CXX_STANDARD 14)

include_directories(
    "D:/MinGW/libs/include"
)

link_directories(
    "D:/MinGW/libs/lib"
)

add_executable(check_wx main.cpp)

target_link_libraries(check_wx

)
