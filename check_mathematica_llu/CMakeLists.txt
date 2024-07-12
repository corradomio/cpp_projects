cmake_minimum_required(VERSION 3.25)
project(check_mathematica_llu)

include_directories(
    "C:/Program Files/Wolfram Research/Mathematica/14.0/SystemFiles/IncludeFiles/C"
    "C:/Program Files/Wolfram Research/Mathematica/14.0/SystemFiles/Links/WSTP/DeveloperKit/Windows-x86-64/CompilerAdditions"
    "D:/Projects/cpp/LibraryLinkUtilities-3.2.0/include"
)

link_directories(
    "C:/Program Files/Wolfram Research/Mathematica/13.3/SystemFiles/Links/WSTP/DeveloperKit/Windows-x86-64/SystemAdditions"
    "D:/Projects/cpp/LibraryLinkUtilities-3.2.0/cmake-build-debug"
)

set(CMAKE_CXX_STANDARD 17)

add_library(check_mathematica_llu SHARED library.cpp)

target_link_libraries(check_mathematica_llu
    wstp64i4
    LLU
)
