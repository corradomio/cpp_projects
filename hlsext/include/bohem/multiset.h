//
// Created by Corrado Mio on 30/09/2020.
//

#ifndef HLSEXT_VECTOR_H
#define HLSEXT_VECTOR_H

#include <multi_set>
#include "allocator.h"

namespace bohem {

    template<typename _Key, typename _Compare = std::less<_Key>>
    struct multi_set : public std::multi_set<_Key, _Compare, bohem::allocator<_Key>> { };

}

#endif //HLSEXT_VECTOR_H
