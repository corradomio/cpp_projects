cmake_minimum_required(VERSION 3.27)
project(commons)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    ./include
)

add_library(commons STATIC
    include/std/real_limits.h
    include/ieee754/real_t.h

    include/intfloat.h
    include/language.h

    include/stdx/exceptions.h
    include/stdx/options.h
    include/stdx/tprintf.h
    include/stdx/array.h
    include/stdx/tryexc.h
    include/stdx/trycatch.h
    include/stdx/memory.h

    src/stdx/options.cpp
    src/stdx/tprintf.cpp
    src/stdx/tryexc.cpp
    src/stdx/memory.cpp
    include/stdx/linalg.h
)

add_custom_command(
    TARGET commons POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/libcommons.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
)
