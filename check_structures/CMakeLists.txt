cmake_minimum_required(VERSION 3.27)
project(check_structures)
include(ExternalProject)

set(CMAKE_CXX_STANDARD 23)

include_directories(
    /libs/include
    ../commons/include
#    /libs/mkl/include
    ./include
)

link_directories(
    /libs/lib
    ../commons/lib
#    /libs/mkl/lib
)


ExternalProject_Add(commons_project
    SOURCE_DIR D:/Projects.github/cpp_projects/commons
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/commons-prefix/src/commons-build/libcommons.a D:/Projects.github/cpp_projects/commons/lib
)


add_executable(check_structures
#    src/main5.cpp
#    src/main6.cpp
#    src/main7.cpp
#    src/main8.cpp
    src/main9.cpp
)

add_executable(test_structures
    tests/test_float64.cpp
)


add_library(catch2_local
    include/stdx/float64/svd.h
)


target_link_libraries(test_structures
    commons
    catch2_local
)

target_link_libraries(check_structures
    commons
    openblas
)

#target_link_libraries(check_mkl
#    mkl_core
#    mkl_intel_thread
#    mkl_rt
#    mkl_tbb_thread
#)

