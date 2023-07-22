cmake_minimum_required(VERSION 3.25)
project(check_mathematica CXX)

set(CMAKE_CPP_STANDARD 17)

include_directories(
    "C:/Program Files/Wolfram Research/Mathematica/13.3/SystemFiles/IncludeFiles/C"
    "C:/Program Files/Wolfram Research/Mathematica/13.3/SystemFiles/Links/WSTP/DeveloperKit/Windows-x86-64/CompilerAdditions"
    "D:/Projects/cpp/LibraryLinkUtilities-3.2.0/include"
)

link_directories(
    "C:/Program Files/Wolfram Research/Mathematica/13.3/SystemFiles/Links/WSTP/DeveloperKit/Windows-x86-64/SystemAdditions"
    "D:/Projects/cpp/LibraryLinkUtilities-3.2.0/cmake-build-debug"
)

add_library(check_mathematica SHARED library.cpp)

target_link_libraries(check_mathematica
    wstp64i4
    LLU
)
