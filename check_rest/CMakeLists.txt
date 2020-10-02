cmake_minimum_required(VERSION 3.17)
project(check_rest)

set(CMAKE_CXX_STANDARD 14)
set(_MSC_VER 1900)

include_directories(
    "D:/MinGW/userlibs/include"
#    "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.27.29110/include"
#    "C:/Program Files (x86)/Windows Kits/10/Include/10.0.18362.0/ucrt"
)

link_directories(
    "D:/MinGW/userlibs/lib"
#    "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.27.29110/lib/onecore/x64"
#    "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.27.29110/lib/x64"
)


add_executable(check_rest main.cpp)

target_link_libraries(check_rest
    gc
    gccpp
    cpprestsdk
    kernel32
)