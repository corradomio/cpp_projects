cmake_minimum_required(VERSION 3.17)
project(check_wsl)

include_directories(
    ~/libs/include
    /mnt/d/Projects.github/cpp_projects/hlsext/include
)

link_directories(
    ~/libs/lib
    /mnt/d/Projects.github/cpp_projects/hlsext/lib
)

set(CMAKE_CXX_STANDARD 14)

add_executable(check_wsl main.cpp)

target_link_libraries(check_wsl
    tbb
)