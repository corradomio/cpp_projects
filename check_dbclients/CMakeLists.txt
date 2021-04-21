cmake_minimum_required(VERSION 3.17)
project(check_dbclients)

set(CMAKE_CXX_STANDARD 14)

include_directories(
    "D:/Programming/mysql-5.7.31-winx64/include"
)

link_directories(
    "D:/Programming/mysql-5.7.31-winx64/lib"
)

add_executable(check_dbclients main.cpp)

target_link_libraries(check_dbclients
    mysqlclient.lib
)