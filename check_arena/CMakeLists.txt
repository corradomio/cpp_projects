cmake_minimum_required(VERSION 3.15)
project(check_arena)

set(CMAKE_CXX_STANDARD 14)

include_directories(
    "D:\\Programming\\CUDA\\v10.1.0\\include"
)

link_directories(
    "D:\\Programming\\CUDA\\v10.1.0\\lib\\x64"
)

add_executable(check_arena main.cpp arena.h)