cmake_minimum_required(VERSION 3.27)
project(check_cublas)

set(CMAKE_CXX_STANDARD 23)
#set(CMAKE_CXX_FLAGS "-Ofast -Wall -Wextra")
#set(CMAKE_CXX_COMPILER "D:/Programming/oneAPI/compiler/latest/bin/icpx.exe")


include_directories(
    "D:/Programming/CUDA/v12.4/include"
)

link_directories(
    "D:/Programming/CUDA/v12.4/bin"
)

add_executable(check_cublas main1.cpp
    cuda/cublas.cpp
    cuda/cublas.h
    main2.cpp
    cuda/vector.cpp
    cuda/matrix.cpp
    main3.cpp
    cuda/array.cpp
    cuda/cublaserr.cpp
    cuda/dot_op.h
    cuda/dot_op.cpp
    cuda/nmf.h
    cuda/nmf.cpp)

target_link_libraries(check_cublas
    cublas64_12
    cublasLt64_12
    cudart64_12
)
