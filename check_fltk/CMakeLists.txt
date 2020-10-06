cmake_minimum_required(VERSION 3.17)
project(check_fltk)

include_directories(
#    ~/libs/include
#    /mnt/d/Projects.github/cpp_projects/hlsext/include
    /libs/include
)

link_directories(
#    ~/libs/lib
#    /mnt/d/Projects.github/cpp_projects/hlsext/lib
    /libs/lib
)

set(CMAKE_CXX_STANDARD 14)

add_executable(check_fltk main.cpp)

target_link_libraries(check_fltk
    tbb
    fltk
    fltk_forms
    fltk_images
#    fltk_jpeg
#    fltk_png
#    X11
#    dl
)
