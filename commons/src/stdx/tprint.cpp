//
// Created by Corrado Mio on 20/03/2024.
//
#include <stdio.h>
#include <time.h>
#include "stdx/tprint.h"

namespace stdx {

    void tprint() {
        time_t t = time(nullptr);
        tm *lt = localtime(&t);
        printf("[%02d:%02d:%02d] ", lt->tm_hour, lt->tm_min, lt->tm_sec);
        fflush(stdout);
    }

}