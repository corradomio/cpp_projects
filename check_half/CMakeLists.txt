cmake_minimum_required(VERSION 3.25)
project(check_half)

include_directories(
    "D:/MinGW/msys64/libs/include"
    "D:/Projects.github/cpp_projects/hlsext/include"
)

link_directories(
    "D:/MinGW/msys64/libs/lib"
)

set(CMAKE_CXX_STANDARD 23)

add_executable(check_half
    main.cpp
    float16_t.h
    minifloat.h
    bfloat16_t.h
    float24.h
    floatx.h
)
