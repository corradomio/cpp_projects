//
// Created by Corrado Mio on 30/09/2020.
//

#ifndef HLSEXT_VECTOR_H
#define HLSEXT_VECTOR_H

#include <set>
#include "allocator.h"

namespace bohem {

    template<typename _Key, typename _Compare = std::less<_Key>>
    struct set : public std::set<_Key, _Compare, bohem::allocator<_Key>> { };

}

#endif //HLSEXT_VECTOR_H