//
// Created by Corrado Mio on 07/06/2024.
//

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
