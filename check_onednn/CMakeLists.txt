cmake_minimum_required(VERSION 3.26)
project(check_onednn)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_onednn main.cpp)

target_link_libraries(check_onednn
    dnnl
)