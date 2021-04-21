cmake_minimum_required(VERSION 3.19)
project(check_dbase)

include_directories(
    #    ~/libs/include
    #    /mnt/d/Projects.github/cpp_projects/hlsext/include
    #    /libs/include
    "D:/MinGW/libs/include"
#    "D:/Programming/pgsql/include"
)

link_directories(
    #    ~/libs/lib
    #    /mnt/d/Projects.github/cpp_projects/hlsext/lib
    #    /libs/lib
    "D:/MinGW/libs/lib"
#    "D:/Programming/pgsql/lib"
)


set(CMAKE_CXX_STANDARD 17)

add_executable(check_dbase main.cpp)

target_link_libraries(check_dbase
    pqxx
    pq
    ws2_32
    mysql.lib
)