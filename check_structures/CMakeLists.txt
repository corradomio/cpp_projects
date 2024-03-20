cmake_minimum_required(VERSION 3.27)
project(check_structures)

set(CMAKE_CXX_STANDARD 23)

include_directories(
    /libs/include
    ./include
    "D:/Programming/mkl/include"
)

link_directories(
    /libs/lib
    "D:/Programming/mkl/lib"
    "D:/Programming/mkl/bin"
)

add_executable(check_structures
    include/stdx/float64/linalg.h
    include/stdx/exceptions.h
    include/stdx/options.h
    src/stdx/options.cpp
    include/stdx/float64/vector.h
    src/stdx/float64/vector.cpp
    include/stdx/float64/vector_op.h
    src/stdx/float64/vector_op.cpp
    include/stdx/float64/arith.h
    include/stdx/float64/matrix.h
    src/stdx/float64/array.cpp
    include/stdx/float64/array.h
    src/stdx/float64/array_op.cpp
    include/stdx/float64/array_op.h
    src/stdx/float64/matrix.cpp
    include/stdx/float64/dot_op.h
    src/stdx/float64/dot_op.cpp
    include/stdx/float64/matrix_op.h
    src/stdx/float64/matrix_op.cpp
    include/stdx/float64/nmf.h
    src/stdx/float64/nmf.cpp
    src/main5.cpp
    src/main6.cpp
    src/stdx/tprintf.cpp
)

add_executable(test_structures
    include/stdx/array.h
    include/stdx/exceptions.h
    include/stdx/options.h
    src/stdx/options.cpp
    include/stdx/float64/vector.h
    src/stdx/float64/vector.cpp
    include/stdx/float64/vector_op.h
    src/stdx/float64/vector_op.cpp
    include/stdx/float64/arith.h
    include/stdx/float64/matrix.h
    src/stdx/float64/array.cpp
    include/stdx/float64/array.h
    src/stdx/float64/array_op.cpp
    include/stdx/float64/array_op.h
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
    include/stdx/tprintf.h
)


target_link_libraries(test_structures
    catch2_local

#    mkl_core
#    mkl_intel_thread
#    mkl_rt
#    mkl_tbb_thread
#    mkl_intel_ilp64
#    mkl_intel_lp64
)

target_link_libraries(check_structures
#    mkl_core
#    mkl_intel_thread
#    mkl_rt
#    mkl_tbb_thread
#    mkl_intel_ilp64
#    mkl_intel_lp64
    openblas
)