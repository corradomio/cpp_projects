cmake_minimum_required(VERSION 3.27)
project(check_cublas)
include(ExternalProject)

set(CMAKE_CXX_STANDARD 23)
#set(CMAKE_CXX_FLAGS "-Ofast -Wall -Wextra")
#set(CMAKE_CXX_COMPILER "D:/Programming/oneAPI/compiler/latest/bin/icpx.exe")


include_directories(
    ./include
    ../commons/include
    /libs/include
    "D:/Programming/CUDA/v12.4/include"
)

link_directories(
    /libs/lib
    ../commons/lib
    "D:/Programming/CUDA/v12.4/bin"
)


ExternalProject_Add(commons_project
    SOURCE_DIR D:/Projects.github/cpp_projects/commons
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/commons-prefix/src/commons-build/libcommons.a D:/Projects.github/cpp_projects/commons/lib
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
    commons
    cublas64_12
    cublasLt64_12
    cudart64_12
)
