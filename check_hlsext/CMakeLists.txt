cmake_minimum_required(VERSION 3.17)
project(check_hlsext)

set(CMAKE_CXX_STANDARD 14)

include_directories(
    ../hlsext/include
)

link_directories(
    ../hlsext/lib
)

add_executable(check_hlsext
    main1.cpp
    main2.cpp
    main3.cpp
    main4.cpp
    ../hlsext/include/stdx/random.h
    ../hlsext/include/stdx/cmath.h
    ../hlsext/include/stdx/containers.h
    ../hlsext/include/stdx/ranges.h
    main5.cpp main6.cpp ../hlsext/include/stdx/params.h main7.cpp)

target_link_libraries(check_hlsext
    hlsext
#    gcmt-lib
)

