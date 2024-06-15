cmake_minimum_required(VERSION 3.28)
project(check_cuda)
include(ExternalProject)

include(FetchContent)
FetchContent_Declare(cuda-api-wrappers_library
    GIT_REPOSITORY https://github.com/eyalroz/cuda-api-wrappers.git
    GIT_TAG v0.6.9 # Replace this with a real available version
    OVERRIDE_FIND_PACKAGE
)

include_directories(
    ./include
    ../commons/include
    /libs/include
    "D:/Programming/CUDA/v12.5/include"
)

link_directories(
    /libs/lib
    ../commons/lib
    "D:/Programming/CUDA/v12.5/bin"
    "D:/Programming/CUDA/v12.5/lib/x64"
)

set(CMAKE_CXX_STANDARD 17)


ExternalProject_Add(commons_project
    SOURCE_DIR D:/Projects.github/cpp_projects/commons
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/commons-prefix/src/commons-build/libcommons.a D:/Projects.github/cpp_projects/commons/lib
)


add_executable(check_cuda
    src/common.cpp
    include/cudacpp/cudacpp.h
    include/cudacpp/cudamem.h
    src/cudacpp/cudacpp.cpp
    src/cudacpp/cudamem.cpp
#    src/main1.cpp
    src/main2.cpp
#    src/main3.cpp
    src/main4.cpp
    src/main5.cpp
    src/main6.cpp
)

target_link_libraries(check_cuda
    commons
#    cudart64_12
#    nvcuda
    cuda
    cudart
)