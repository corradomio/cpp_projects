cmake_minimum_required(VERSION 3.17)
project(check_postgres)

set(CMAKE_CXX_STANDARD 20)

include_directories("D:/MinGW/libs/include")
link_directories("D:/MinGW/libs/lib")

add_executable(check_postgres main.cpp)

target_link_libraries(check_postgres
    pqxx
    pq
    ws2_32
)