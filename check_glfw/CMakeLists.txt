cmake_minimum_required(VERSION 3.27)
project(check_glfw)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_glfw
    main1.cpp
)

target_link_libraries(check_glfw
    glfw3
    opengl32
    gdi32
)
