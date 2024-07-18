cmake_minimum_required(VERSION 3.28)
project(check_linalg)

set(CMAKE_CXX_STANDARD 17)
include(ExternalProject)


include_directories(
    ./include
    ../commons/include
    /libs/include
)

link_directories(
    /libs/lib
    ../commons/lib
)

set(CMAKE_CXX_STANDARD 23)


ExternalProject_Add(commons_project
    SOURCE_DIR D:/Projects.github/cpp_projects/commons
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/commons-prefix/src/commons-build/libcommons.a D:/Projects.github/cpp_projects/commons/lib
)

add_executable(check_linalg
    include/stdx/float64/arith.h
    include/stdx/float64/array.h
    include/stdx/float64/array_op.h
    include/stdx/float64/dot_op.h
    include/stdx/float64/matrix.h
    include/stdx/float64/matrix_op.h
    include/stdx/float64/vector.h
    include/stdx/float64/vector_op.h
    include/stdx/float64/nmf.h
    include/stdx/float64/svd.h

    src/stdx/float64/array.cpp
    src/stdx/float64/array_op.cpp
    src/stdx/float64/dot_op.cpp
    src/stdx/float64/matrix.cpp
    src/stdx/float64/matrix_op.cpp
    src/stdx/float64/vector.cpp
    src/stdx/float64/vector_op.cpp
    src/stdx/float64/nmf.cpp
    src/stdx/float64/svd.cpp

    src/main.cpp
)

target_link_libraries(check_linalg
    commons
#    lapack
#    lapacke
    openblas
)