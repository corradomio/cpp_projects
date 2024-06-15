cmake_minimum_required(VERSION 3.28)
project(check_cuda_cu CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_library(check_cuda_cu STATIC library.cu)

set_target_properties(check_cuda_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
