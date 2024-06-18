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
    src/stdx/number/i32/numbers.cpp
    include/stdx/number/i32/numbers.h
    include/stdx/number/i128/numbers.h
    src/stdx/number/i128/numbers.cpp
)

target_link_libraries(check_number_system
    commons
)
