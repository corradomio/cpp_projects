cmake_minimum_required(VERSION 3.27)
project(check_structures)

set(CMAKE_CXX_STANDARD 23)

include_directories(
    /libs/include
    ./include
)

link_directories(
    /libs/lib
)

add_executable(check_structures
        include/stdx//exceptions.h
        include/stdx//options.h
        src/stdx/options.cpp
        include/stdx//float64/vector.h
        src/stdx/float64/vector.cpp
        include/stdx//float64/vector_op.h
        src/stdx/float64/vector_op.cpp
        include/stdx//arith/arith.h
        include/stdx//float64/matrix.h
        src/stdx/float64/array.cpp
        include/stdx//float64/array.h
        src/stdx/float64/array_op.cpp
        include/stdx//float64/array_op.h
        src/stdx/float64/matrix.cpp
        include/stdx//float64/dot_op.h
        src/stdx/float64/dot_op.cpp
        include/stdx//float64/matrix_op.h
        src/stdx/float64/matrix_op.cpp
        include/stdx//float64/transpose.h
        src/stdx/float64/transpose.cpp
        include/stdx//float64/matrix_factorization.h
        src/stdx/float64/matrix_factorization.cpp
        src/main5.cpp
        include/stdx//ref/map.h
)

add_executable(test_structures
        include/stdx//array.h
        include/stdx//exceptions.h
        include/stdx//options.h
        src/stdx/options.cpp
        include/stdx//float64/vector.h
        src/stdx/float64/vector.cpp
        include/stdx//float64/vector_op.h
        src/stdx/float64/vector_op.cpp
        include/stdx//arith/arith.h
        include/stdx//float64/matrix.h
        src/stdx/float64/array.cpp
        include/stdx//float64/array.h
        src/stdx/float64/array_op.cpp
        include/stdx//float64/array_op.h
        src/stdx/float64/matrix.cpp
        include/stdx/float64/dot_op.h
        src/stdx/float64/dot_op.cpp
        include/stdx/float64/matrix_op.h
        src/stdx/float64/matrix_op.cpp
    tests/test_float64.cpp
)


add_library(catch2_local
    catch2/catch_amalgamated.hpp
    catch2/catch_amalgamated.cpp
        include/stdx/ref/map.h
)


target_link_libraries(test_structures
    catch2_local
)