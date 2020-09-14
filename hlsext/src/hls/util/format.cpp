//
// Created by Corrado Mio on 17/10/2015.
//

#include <stdarg.h>
#include <stdio.h>
#include "../../../include/hls/util/format.hpp"

using namespace hls::util;

std::string hls::util::format(const char* fmt, ...)
{
    char buffer[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buffer, 1024, fmt, ap);
    return std::string(buffer);
}
