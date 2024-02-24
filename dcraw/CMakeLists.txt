cmake_minimum_required(VERSION 3.27)
project(dcraw)

set(CMAKE_CXX_STANDARD 17)
set(NODEPS 1)

add_executable(dcraw
    dcraw.c
)

target_link_libraries(dcraw
    tiff
    ws2_32
)