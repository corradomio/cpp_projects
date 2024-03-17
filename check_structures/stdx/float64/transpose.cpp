//
// Created by Corrado Mio on 09/03/2024.
//
#include <iostream>
#include "transpose.h"

namespace stdx::float64 {

    tr_t tr(const matrix_t& m) { return tr_t(m); }

    void print(const tr_t& m) {
        size_t nr = m.rows();
        size_t nc = m.cols();

        std::cout << "[" << std::endl;
        for(int i=0; i<nr; ++i) {
            std::cout << "  [ ";
            for(int j=0; j<nc; ++j)
                std::cout << m[i, j] << " ";
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }
}