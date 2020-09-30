//
// Created by Corrado Mio on 30/09/2020.
//

#ifndef HLSEXT_VECTOR_H
#define HLSEXT_VECTOR_H

#include <list>
#include "allocator.h"

namespace bohem {

    template<typename _Tp>
    struct list : public std::list<_Tp, bohem::allocator<_Tp>> { };

}

#endif //HLSEXT_VECTOR_H
