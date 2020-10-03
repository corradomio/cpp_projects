cmake_minimum_required(VERSION 3.17)
project(check_asio)

set(CMAKE_CXX_STANDARD 14)

add_compile_definitions(BOOST_LOG_DYN_LINK=1)

add_executable(check_asio main.cpp)

target_link_libraries(check_asio ws2_32 mswsock)