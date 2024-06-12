//
// Created by Corrado Mio on 08/06/2024.
//

#ifndef CHECK_CUDA_COMMON_H
#define CHECK_CUDA_COMMON_H

#include "cudacpp/cudacpp.h"
#include "cudacpp/cudamem.h"


extern double sum(cudacpp::array_t<float> A, cudacpp::array_t<float> B, float c);
extern double sum(cudacpp::array_t<float> C);

// extern void tprintf(const char *__format, ...);
// extern void check(CUresult res);

#endif //CHECK_CUDA_COMMON_H
