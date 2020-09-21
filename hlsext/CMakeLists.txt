cmake_minimum_required(VERSION 3.2)
project(hlsext)

include_directories(
        "./include"
)
link_directories(
    "./lib"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
#    src/cstd/realpath.c
#    src/cstd/malloc-2.8.6.c
#    src/cstd/malloc-2.8.6.h

    src/hls/app/main.cpp
    src/hls/util/format.cpp
    src/hls/lang/exception.cpp
    src/hls/lang/trycatch.cpp
    src/hls/memory/alloc.cpp
    src/hls/memory/global.cpp
    src/hls/memory/arena.cpp
    src/hls/memory/buffer.cpp
    include/stdx/containers.h
    include/stdx/ref_map.h
    include/stdx/ref_vector.h
    include/stdx/ref_set.h
    include/stdx/ref_unordered_map.h
    include/stdx/ref_unordered_set.h
    include/boostx/date_time_op.h
)

add_library(hlsext ${SOURCE_FILES})

#add_executable(check main2.cpp)
#target_link_libraries(check
#    hlsext
#)

