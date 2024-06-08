//
// Created by Corrado Mio on 07/06/2024.
//

#include <stdio.h>
#include <time.h>
#include <stdarg.h>
#include <cuda.h>
#include "cudacpp/cudamem.h"

using namespace cudacpp;


double sum(array_t<float> A, array_t<float> B, float c) {
    size_t n = A.size();
    double r = 0;;
    for(size_t i=0; i<n; ++i)
        r += A[i] +B[i] + c;
    return r;
}

double sum(array_t<float> C) {
    size_t n = C.size();
    double r = 0;;
    for(size_t i=0; i<n; ++i)
        r += C[i];
    return r;
}

void tprintf(const char *__format, ...) {
    time_t t = time(nullptr);
    tm *lt = localtime(&t);
    printf("[%02d:%02d:%02d] ", lt->tm_hour, lt->tm_min, lt->tm_sec);
    va_list argv; va_start( argv, __format );
    vfprintf( stdout, __format, argv );
    va_end( argv );
    fflush(stdout);
}
