cmake_minimum_required(VERSION 3.27)
project(check_cudaexe CUDA)

set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -cuda")

include_directories(
    "D:/Programming/CUDA/cuda-samples-12.4/Common"
)

add_executable(check_cudaexe
    main.cu
    vectorAdd.cu
)

set_target_properties(check_cudaexe PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
