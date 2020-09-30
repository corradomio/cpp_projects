//
// Created by Corrado Mio on 30/09/2020.
//

#ifndef HLSEXT_VECTOR_H
#define HLSEXT_VECTOR_H

#include <map>
#include "allocator.h"

namespace bohem {

    template<typename _Key, typename _Tp, typename _Compare = std::less<_Key>>
    struct map : public std::map<_Key, _Tp, _Compare, bohem::allocator<std::pair<const _Key, _Tp>> > { };

}

#endif //HLSEXT_VECTOR_H
