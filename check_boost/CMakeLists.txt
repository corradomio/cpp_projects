cmake_minimum_required(VERSION 3.17)
project(check_boost)

set(CMAKE_CXX_STANDARD 14)

add_compile_definitions(BOOST_LOG_DYN_LINK=1)

add_executable(check_boost main.cpp)

target_link_libraries(check_boost
    boost_log-mt
)