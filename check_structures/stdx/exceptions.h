//
// Created by Corrado Mio on 08/03/2024.
//

#include <stdexcept>

#ifndef STDX_EXCEPTIONS_H
#define STDX_EXCEPTIONS_H

namespace stdx {

    struct bad_dimensions : public std::runtime_error {
        bad_dimensions(): std::runtime_error("Incompatible dimensions") {}
    };


}

#endif //STDX_EXCEPTIONS_H
