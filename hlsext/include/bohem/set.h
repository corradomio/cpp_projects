//
// Created by Corrado Mio on 30/09/2020.
//

#ifndef HLSEXT_VECTOR_H
#define HLSEXT_VECTOR_H

#include <vector>
#include "allocator.h"

namespace bohem {

    template<typename T>
    struct vector : std::vector<T, bohem::allocator<T>> { };

}

#endif //HLSEXT_VECTOR_H
