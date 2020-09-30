//
// Created by Corrado Mio on 30/09/2020.
//

#ifndef HLSEXT_VECTOR_H
#define HLSEXT_VECTOR_H

#include <set>
#include "allocator.h"

namespace bohem {

    template<typename T, typename Compare>
    struct set : std::set<T, Compare, bohem::allocator<T>> { };

}

#endif //HLSEXT_VECTOR_H
