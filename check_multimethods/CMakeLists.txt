cmake_minimum_required(VERSION 3.20)
project(check_multimethods)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    "D:/MinGW/libs/include"
    "../hlsext/include"
)

link_directories(
    "D:/MinGW/libs/lib"
    ../hlsext/lib
)

add_executable(check_multimethods main.cpp)

target_link_libraries(check_multimethods
#    yomm2
)