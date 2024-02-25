cmake_minimum_required(VERSION 3.27)
project(check_linalg)

include_directories("D:/MinGW/libs/include")
link_directories("D:/MinGW//libs/lib")

# C++23 accepts operator[](int, int)
#
set(CMAKE_CXX_STANDARD 23)

add_library(linalg
    linalg/linalg.h
    linalg/vec.cpp
    linalg/mat.cpp
    linalg/check.cpp
)

add_library(catch2v3
    catch_amalgamated.cpp
)

add_executable(check_linalg
    main.cpp
)

target_link_libraries(check_linalg
    linalg
)

add_executable(test_linalg
    tests_run.cpp
    tests/test_vec.cpp
#    catch_amalgamated.cpp
)

target_link_libraries(test_linalg
    linalg
    catch2v3
#    Catch2
#    Catch2Main
)

