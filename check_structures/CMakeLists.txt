cmake_minimum_required(VERSION 3.27)
project(check_structures)

set(CMAKE_CXX_STANDARD 23)

include_directories(
    /libs/include
)

link_directories(
    /libs/lib
)

add_executable(check_structures
        stdx/array.h
main3.cpp
        stdx/linalg/vector.h
        stdx/linalg/vector_op.h
        stdx/linalg/matrix_op.h
        stdx/linalg/matrix.h
        stdx/linalg/dot_op.h
        stdx/linalg/print_op.h
)

add_executable(test_structures
    tests/test_vec.cpp
        stdx/array.h
        stdx/linalg/vector.h
        stdx/linalg/vector_op.h
        stdx/linalg/matrix.h
        stdx/linalg/matrix_op.h
)


add_library(catch2_local
    catch2/catch_amalgamated.hpp
    catch2/catch_amalgamated.cpp
)


target_link_libraries(test_structures
    catch2_local
)