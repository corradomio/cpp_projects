//
// Created by Corrado Mio on 02/06/2024.
//

#include <stdx/language.h>
#include "../include/cerebras.h"

namespace cerebras {

    wafer_scale_engine_t::wafer_scale_engine_t(size_t rows, size_t cols)
    : rows(rows), cols(cols) {

    }

    wafer_scale_engine_t WSE{128, 128};

}