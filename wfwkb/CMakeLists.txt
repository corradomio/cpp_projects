cmake_minimum_required(VERSION 3.25)
project(wfwkb)

include_directories(
    "D:/MinGW/msys64/libs/include"
    "C:/Program Files/Wolfram Research/Mathematica/13.3/SystemFiles/IncludeFiles/C"
    "C:/Program Files/Wolfram Research/Mathematica/13.3/SystemFiles/Links/WSTP/DeveloperKit/Windows-x86-64/CompilerAdditions"
    "D:/Projects/cpp/LibraryLinkUtilities-3.2.0/include"
)

link_directories(
    "D:/MinGW/msys64/libs/lib"
    "C:/Program Files/Wolfram Research/Mathematica/13.3/SystemFiles/Links/WSTP/DeveloperKit/Windows-x86-64/SystemAdditions"
    "D:/Projects/cpp/LibraryLinkUtilities-3.2.0/cmake-build-debug"
)

set(CMAKE_CXX_STANDARD 17)

add_library(wfwkb SHARED library.cpp LLULogger.h)

target_link_libraries(wfwkb
    wstp64i4
    LLU
    spdlog
)
