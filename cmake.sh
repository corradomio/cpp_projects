#!/bin/bash
echo "cmake -G'MSYS Makefiles' -DCMAKE_INSTALL_PREFIX=/userlibs/temp $*"
#/mingw64/bin/cmake -G"MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=/usr/local $*
#/mingw64/bin/cmake -G"MinGW Makefiles" -DCMAKE_INSTALL_PREFIX=/usr/local $*
/mingw64/bin/cmake -G'MSYS Makefiles' -DCMAKE_INSTALL_PREFIX=/userlibs/temp $*
