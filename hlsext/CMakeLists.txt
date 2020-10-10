cmake_minimum_required(VERSION 3.2)
project(hlsext)

include_directories(
    ./include
)
link_directories(
    ./lib
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
#    src/cstd/realpath.c
#    src/cstd/malloc-2.8.6.c
#    src/cstd/malloc-2.8.6.h

    include/boostx/date_time_op.h
    include/bohem/vector
    include/bohem/string
    include/bohem/containers
    include/ref/map.h
    include/ref/vector.h
    include/ref/set.h
    include/ref/unordered_map.h
    include/ref/unordered_set.h
    include/stdx/bag.h
    include/stdx/containers.h
    include/hls/util/properties.h

    src/hls/app/main.cpp
    src/stdx/exception.cpp
    src/stdx/trycatch.cpp
    src/hls/memory/alloc.cpp
    src/hls/memory/global.cpp
    src/hls/memory/arena.cpp
    src/hls/memory/buffer.cpp
    src/hls/util/properties_utils.h
    src/hls/util/properties.cpp
    src/hls/util/properties_utils.cpp
)

add_library(hlsext ${SOURCE_FILES})
