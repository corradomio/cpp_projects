cmake_minimum_required(VERSION 3.17)
project(check_gui)

include_directories(
#    ~/libs/include
#    /mnt/d/Projects.github/cpp_projects/hlsext/include
#    /libs/include
    "d:/MinGW/libs/include"
)

link_directories(
#    ~/libs/lib
#    /mnt/d/Projects.github/cpp_projects/hlsext/lib
#    /libs/lib
    "d:/MinGW/libs/lib"
)

set(CMAKE_CXX_STANDARD 14)

add_executable(check_gui main_nana.cpp main_fltk.cpp main_wxtk.cpp)

target_link_libraries(check_gui
    tbb
    fltk
    fltk_forms
    fltk_images
    comctl32.lib
    nana
    wx_mswud-3.1.dll
#    fltk_jpeg
#    fltk_png
#    X11
#    dl
)
