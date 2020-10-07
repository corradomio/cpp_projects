cmake_minimum_required(VERSION 3.17)
project(check_wsl)

include_directories(
    /mnt/d/Projects.github/cpp_projects/hlsext/include
    /home/ubuntu/libs/include
)

link_directories(
    /mnt/d/Projects.github/cpp_projects/hlsext/lib
    /home/ubuntu/libs/lib
)

set(CMAKE_CXX_STANDARD 14)

add_executable(check_wsl main.cpp)

target_link_libraries(check_wsl
    tbb
)