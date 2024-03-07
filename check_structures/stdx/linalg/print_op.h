//
// Created by Corrado Mio on 07/03/2024.
//

#ifndef STDX_PRINT_OP_H
#define STDX_PRINT_OP_H

#include <iostream>
#include "vector.h"
#include "matrix.h"


namespace stdx::linalg {

    template<typename T>
    void print(const vector_t<T>& v) {
        std::cout << "[";
        for (int i=0; i<v.size(); ++i)
            std::cout << " " << v[i];
        std::cout << " ]" << std::endl;
    }

    template<typename T>
    void print(const matrix_t<T>& m) {
        std::cout << "[" << std::endl;
        for (int i=0; i<m.rows(); ++i) {
            std::cout << " ";
            for (int j=0; j<m.cols(); ++j)
                std::cout << " " << m[i,j];
            std::cout << std::endl;
        }
        std::cout << "]" << std::endl;
    }
}

#endif //STDX_PRINT_OP_H
