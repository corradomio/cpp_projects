cmake_minimum_required(VERSION 3.27)
project(check_structures)

set(CMAKE_CXX_STANDARD 23)

include_directories(
    /libs/include
    ./include
    "D:/MinGW/libs/mkl/include"
)

link_directories(
    /libs/lib
    "D:/MinGW/libs/mkl/lib"
)

add_executable(check_structures
    include/stdx/exceptions.h
    include/stdx/float64/arith.h
    include/stdx/float64/array.h
    include/stdx/float64/array_op.h
    include/stdx/float64/dot_op.h
    include/stdx/float64/linalg.h
    include/stdx/float64/matrix.h
    include/stdx/float64/matrix_op.h
    include/stdx/float64/nmf.h
    include/stdx/float64/vector.h
    include/stdx/float64/vector_op.h
    include/stdx/options.h
    src/main5.cpp
    src/main6.cpp
    src/main7.cpp
    src/stdx/float64/array.cpp
    src/stdx/float64/array_op.cpp
    src/stdx/float64/dot_op.cpp
    src/stdx/float64/matrix.cpp
    src/stdx/float64/matrix_op.cpp
    src/stdx/float64/nmf.cpp
    src/stdx/float64/vector.cpp
    src/stdx/float64/vector_op.cpp
    src/stdx/options.cpp
    src/stdx/tprint.cpp
)

add_executable(test_structures
    include/stdx/array.h
    include/stdx/exceptions.h
    include/stdx/float64/arith.h
    include/stdx/float64/array.h
    include/stdx/float64/array_op.h
    include/stdx/float64/dot_op.h
    include/stdx/float64/matrix.h
    include/stdx/float64/matrix_op.h
    include/stdx/float64/vector.h
    include/stdx/float64/vector_op.h
    include/stdx/options.h
    src/stdx/float64/array.cpp
    src/stdx/float64/array_op.cpp
    src/stdx/float64/dot_op.cpp
    src/stdx/float64/matrix.cpp
    src/stdx/float64/matrix_op.cpp
    src/stdx/float64/vector.cpp
    src/stdx/float64/vector_op.cpp
    src/stdx/options.cpp
    tests/test_float64.cpp
)


add_executable(check_mkl
    include/stdx/exceptions.h
    include/stdx/float64/arith.h
    include/stdx/float64/array.h
    include/stdx/float64/array_op.h
    include/stdx/float64/dot_op.h
    include/stdx/float64/linalg.h
    include/stdx/float64/matrix.h
    include/stdx/float64/matrix_op.h
    include/stdx/float64/nmf.h
    include/stdx/float64/vector.h
    include/stdx/float64/vector_op.h
    include/stdx/options.h
    src/stdx/float64/array.cpp
    src/stdx/float64/array_op.cpp
    src/stdx/float64/dot_op.cpp
    src/stdx/float64/matrix.cpp
    src/stdx/float64/matrix_op.cpp
    src/stdx/float64/nmf.cpp
    src/stdx/float64/vector.cpp
    src/stdx/float64/vector_op.cpp
    src/stdx/options.cpp
    src/stdx/tprint.cpp
    src/main_mkl.cpp
)


add_library(catch2_local
    catch2/catch_amalgamated.hpp
    catch2/catch_amalgamated.cpp
)


target_link_libraries(test_structures
    catch2_local
)

target_link_libraries(check_structures
    openblas
)


target_link_libraries(check_mkl
    mkl_core
    mkl_intel_thread
    mkl_rt
    mkl_tbb_thread
#    mkl_intel_ilp64
#    mkl_intel_lp64
)