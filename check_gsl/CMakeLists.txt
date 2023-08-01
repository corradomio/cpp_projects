cmake_minimum_required(VERSION 3.26)
project(check_gsl)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_gsl
    main.cpp
)

target_link_libraries(check_gsl
    gsl
)
