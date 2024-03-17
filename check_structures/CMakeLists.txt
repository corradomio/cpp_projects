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
    main3.cpp
    main4.cpp
    stdx/array.h
    stdx/exceptions.h
    stdx/float64/vector.h
    stdx/float64/vector.cpp
    stdx/float64/vector_op.h
    stdx/float64/vector_op.cpp
    stdx/arith.h
    stdx/float64/matrix.h
    stdx/float64/array.cpp
    stdx/float64/array.h
    stdx/float64/array_op.cpp
    stdx/float64/array_op.h
    stdx/float64/matrix.cpp
    stdx/float64/dot_op.h
    stdx/float64/dot_op.cpp
    stdx/float64/matrix_op.h
    stdx/float64/matrix_op.cpp
    stdx/float64/transpose.h
    stdx/float64/transpose.cpp
)

add_executable(test_structures
    stdx/array.h
    stdx/exceptions.h
    stdx/float64/vector.h
    stdx/float64/vector.cpp
    stdx/float64/vector_op.h
    stdx/float64/vector_op.cpp
    stdx/arith.h
    stdx/float64/matrix.h
    stdx/float64/array.cpp
    stdx/float64/array.h
    stdx/float64/array_op.cpp
    stdx/float64/array_op.h
    stdx/float64/matrix.cpp
    stdx/float64/dot_op.h
    stdx/float64/dot_op.cpp
    stdx/float64/matrix_op.h
    stdx/float64/matrix_op.cpp
    tests/test_float64.cpp
)


add_library(catch2_local
    catch2/catch_amalgamated.hpp
    catch2/catch_amalgamated.cpp
)


target_link_libraries(test_structures
    catch2_local
)