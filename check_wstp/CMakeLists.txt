cmake_minimum_required(VERSION 3.25)
project(check_wstp)

include_directories(
    "C:/Program Files/Wolfram Research/Mathematica/13.3/SystemFiles/Links/WSTP/DeveloperKit/Windows-x86-64/CompilerAdditions"
)
link_directories(
    "C:/Program Files/Wolfram Research/Mathematica/13.3/SystemFiles/Links/WSTP/DeveloperKit/Windows-x86-64/SystemAdditions"
)

set(CMAKE_CXX_STANDARD 17)

add_executable(check_wstp main.cpp)

target_link_libraries(check_wstp
    wstp64i4
)
