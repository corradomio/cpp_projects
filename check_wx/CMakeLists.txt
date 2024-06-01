cmake_minimum_required(VERSION 3.21)
project(check_wx)

set(CMAKE_CXX_STANDARD 14)

include_directories(
    "D:/MinGW/libs/wx-3.2/include"
)

link_directories(
    "D:/MinGW/libs/wx-3.2/lib"
)

add_executable(check_wx main.cpp)

target_link_libraries(check_wx
    wx_baseu-3.2
#    wx_baseu_net-3.2
    wx_baseu_xml-3.2
    wx_mswu_adv-3.2
    wx_mswu_aui-3.2
    wx_mswu_core-3.2
    wx_mswu_gl-3.2
    wx_mswu_html-3.2
    wx_mswu_media-3.2
    wx_mswu_propgrid-3.2
    wx_mswu_qa-3.2
    wx_mswu_ribbon-3.2
    wx_mswu_richtext-3.2
    wx_mswu_stc-3.2
    wx_mswu_webview-3.2
    wx_mswu_xrc-3.2
)
