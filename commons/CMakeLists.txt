cmake_minimum_required(VERSION 3.27)
project(commons)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    ./include
)

add_library(commons STATIC
    include/stdx/exceptions.h
    include/stdx/language.h
    include/stdx/options.h
    include/stdx/tprint.h
    include/stdx/array.h
    include/stdx/tryexc.h
    include/stdx/trycatch.h
    src/stdx/options.cpp
    src/stdx/tprint.cpp
    src/stdx/tryexc.cpp
)
