#include <stdarg.h>
#include <stdarg.h>
#include "../../../include/hls/math/tensor.hpp"

using namespace hls::math;

tensor_t::tensor_t()
: rank(0), length(1)
{

}

void tensor_t::_init(const size_t* dim)
{
    size_t i = 0;

    const_cast<size_t&>(this->length) = 1;

    for(i=0; dim[i]>0; ++i)
    {
        const_cast<size_t*>(this->size)[i] = dim[i];
        const_cast<size_t&>(this->length) *= dim[i];
    }
    const_cast<size_t&>(this->rank) = i;
}

size_t tensor_t::_at(const size_t* index) const
{
    size_t at = 0;
    for(int i = 0; i < this->rank; ++i)
        at = at*this->size[i] + index[i];
    return at;
}
