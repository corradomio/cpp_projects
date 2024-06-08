cmake_minimum_required(VERSION 3.28)
project(check_cuda)


include_directories(
    ./include
    /libs/include
    "D:/Programming/CUDA/v12.2/include"
)

link_directories(
    /libs/lib
    "D:/Programming/CUDA/v12.2/bin"
    "D:/Programming/CUDA/v12.2/lib/x64"
)


set(CMAKE_CXX_STANDARD 17)

add_executable(check_cuda
    src/common.cpp
    include/cudacpp.h
    include/cudamem.h
    src/cudacpp.cpp
    src/cudamem.cpp
#    src/main1.cpp
    src/main2.cpp
#    src/main3.cpp
    src/main4.cpp
)

# nvcuda.dll for GNU c++
# cuda.lib   for VS  c++

target_link_libraries(check_cuda
#    cudart64_12
#    nvcuda
    cuda
)