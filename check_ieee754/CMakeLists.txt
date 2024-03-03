cmake_minimum_required(VERSION 3.26)
project(check_ieee754)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_ieee754
    main.cpp
    real_t.h
    real_limits.h
)


add_executable(test_ieee754
    tests/test_ieee754.cpp
    real_t.h
    real_limits.h
)


add_library(catch2_local
    catch2/catch_amalgamated.cpp
    catch2/catch_amalgamated.hpp
)


target_link_libraries(test_ieee754
    catch2_local
)