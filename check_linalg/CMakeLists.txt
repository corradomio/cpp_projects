cmake_minimum_required(VERSION 3.28)
project(check_linalg)

set(CMAKE_CXX_STANDARD 17)
include(ExternalProject)


include_directories(
    ./include
    ../commons/include
    /libs/include
)

link_directories(
    /libs/lib
    ../commons/lib
)

set(CMAKE_CXX_STANDARD 23)


ExternalProject_Add(commons_project
    SOURCE_DIR D:/Projects.github/cpp_projects/commons
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/commons-prefix/src/commons-build/libcommons.a D:/Projects.github/cpp_projects/commons/lib
)

add_executable(check_linalg
    main.cpp
)
