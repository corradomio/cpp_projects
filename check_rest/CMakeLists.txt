cmake_minimum_required(VERSION 3.17)
project(check_rest)

set(CMAKE_CXX_STANDARD 14)
set(_MSC_VER 1900)


add_compile_definitions(BOOST_LOG_DYN_LINK=1)

find_package(Boost REQUIRED COMPONENTS
    log
    system
    program_options
    filesystem
    date_time
    context
    coroutine
    chrono)

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


add_executable(check_rest
#    main1.cpp
#    main2.cpp
    main3.cpp
    main4.cpp main5.cpp)

target_link_libraries(check_rest
    served.dll

    boost_system-mt.dll
    ws2_32
    mswsock

#    restc-cpp
#    ws2_32
#    mswsock
#    ssl
#    crypto
#    z
#    bcrypt
#    ${Boost_LIBRARIES}
)