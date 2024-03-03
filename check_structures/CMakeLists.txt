cmake_minimum_required(VERSION 3.27)
project(check_structures)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_structures
array.h
main3.cpp
        vector.h
        vector_op.h
        matrix_op.h
    matrix.h
)

add_executable(test_structures
    tests/test_vec.cpp
    array.h
    vector.h
    vector_op.h
    matrix.h
    matrix_op.h
)


add_library(catch2_local
    catch2/catch_amalgamated.hpp
    catch2/catch_amalgamated.cpp
)


target_link_libraries(test_structures
    catch2_local
)