cmake_minimum_required(VERSION 3.17)
project(check_tensor)

set(CMAKE_CXX_STANDARD 17)


include_directories(
    ./include
    ../commons/include
    /libs/include
)

link_directories(
    /libs/lib
    ../commons/lib
)

add_executable(check_tensor
    stdx/tensor.cpp
    main.cpp
    stdx/tensor.cpp
)