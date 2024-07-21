cmake_minimum_required(VERSION 3.26)
project(check_onednn)

include_directories(
    "D:/MinGW/libs/onednn/include"
)

link_directories(
    "D:/MinGW/libs/onednn/lib"
)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_onednn main.cpp)

target_link_libraries(check_onednn
    dnnl
)