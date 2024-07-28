cmake_minimum_required(VERSION 3.30)
project(check_nlp)

include_directories(
    "D:\\Projects.github\\cpp_projects\\commons\\include"
)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_nlp
    main.cpp
    text.cpp
    ../commons/src/stdx/strings.cpp
)
