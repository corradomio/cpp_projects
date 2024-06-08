//
// Created by Corrado Mio on 24/03/2024.
//

#ifndef CUDA_NMF_H
#define CUDA_NMF_H

#include "cublas.h"
#include <stdx/options.h>

namespace cuda {

    std::tuple<matrix_t, matrix_t> nmf(const matrix_t& M, size_t k, const stdx::options_t& opts);
}

#endif //CUDA_NMF_H
