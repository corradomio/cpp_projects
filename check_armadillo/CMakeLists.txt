cmake_minimum_required(VERSION 3.27)
project(check_armadillo)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_armadillo
    main.cpp)

target_link_libraries(check_armadillo
    openblas
    armadillo
)
