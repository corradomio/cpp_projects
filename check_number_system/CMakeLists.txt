cmake_minimum_required(VERSION 3.28)
project(check_number_system)

include_directories(
    /libs/include
    ../commons/include
    ./include
)

link_directories(
    /libs/lib
    ../commons/lib
)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_number_system
    main.cpp
    src/stdx/numbers.cpp
    include/stdx/numbers.h
)

target_link_libraries(check_number_system
    commons
)
