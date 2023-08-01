cmake_minimum_required(VERSION 3.25)
project(check_half)

add_compile_options(-masm=intel)

include_directories(
    "D:/MinGW/msys64/libs/include"
    "D:/Projects.github/cpp_projects/hlsext/include"
)

link_directories(
    "D:/MinGW/msys64/libs/lib"
)

set(CMAKE_CXX_STANDARD 23)

add_executable(check_half
    catch_amalgamated.cpp
    catch_amalgamated.hpp
    main.cpp
    float16_t.h
    minifloat.h
    bfloat16_t.h
    float24.h
    floatx.h
    ieee754.h
    ieee754.cpp 
    ieee754std.h
    all_tests.cpp
)
