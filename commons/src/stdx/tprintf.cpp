//
// Created by Corrado Mio on 20/03/2024.
//
#include <stdio.h>
#include <time.h>
#include <stdarg.h>
#include <cstdio>
#include "stdx/tprintf.h"

namespace stdx {

    void tprintf(const char *__format, ...) {
        time_t t = time(nullptr);
        tm *lt = localtime(&t);
        printf("[%02d:%02d:%02d] ", lt->tm_hour, lt->tm_min, lt->tm_sec);

        va_list args;
        va_start(args, __format);
        vprintf(__format, args);
        va_end(args);

        fflush(stdout);
    }

}