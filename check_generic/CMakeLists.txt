cmake_minimum_required(VERSION 3.17)
project(check_generic)

include_directories(
    "D:/MinGW/libs/include"
    "D:/Projects.github/cpp_projects/hlsext/include"
)

link_directories(
    "D:/MinGW/libs/lib"
    "D:/Projects.github/cpp_projects/hlsext/lib"
)

set(CMAKE_CXX_STANDARD 14)

add_executable(check_generic
#    main2.cpp
#    main3.cpp
#    main4.cpp
#    main5.cpp
        main7.cpp main8.cpp main.cpp)

target_link_libraries(check_generic
#    pqxx
#    pq
#    ws2_32
#    hlsext
#    gc
)
