cmake_minimum_required(VERSION 3.27)
project(check_lowrank)

set(CMAKE_CXX_STANDARD 23)

add_executable(check_lowrank main.cpp
    linalg.cpp
    linalg.h
    refc.h)
