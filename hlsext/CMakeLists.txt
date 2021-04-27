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

    include/boost/date_time/date_time_op.h
    include/bohem/vector
    include/bohem/string
    include/bohem/containers
    include/ref/map
    include/ref/vector
    include/ref/set
    include/ref/unordered_map
    include/ref/unordered_set
    include/stdx/bag.h
    include/stdx/containers.h
    include/stdx/properties.h
    include/stdx/strings.h
    include/cpp/syntax.h

    src/hls/app/main.cpp
    src/stdx/exception.cpp
    src/stdx/trycatch.cpp
    src/hls/memory/alloc.cpp
    src/hls/memory/global.cpp
    src/hls/memory/arena.cpp
    src/hls/memory/buffer.cpp
    src/hls/util/properties_utils.h
    src/stdx/properties.cpp
    src/hls/util/properties_utils.cpp
    src/stdx/strings.cpp
)

add_library(hlsext ${SOURCE_FILES})
