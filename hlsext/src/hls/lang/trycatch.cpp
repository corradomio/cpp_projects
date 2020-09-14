/*
 * File:   trycatch.cpp
 * Author: ...
 *
 * Created on April 24, 2013, 10:48 AM
 */

#include "../../../include/hls/lang/trycatch.hpp"

using namespace hls::lang;

// --------------------------------------------------------------------------
// try_t
// --------------------------------------------------------------------------

try_t::try_t(const char* fn, int l, const char* fu)
: block_t(fn, l, fu), _is_finally(false)
{

}

try_t::try_t()
: block_t(), _is_finally(false)
{

}

try_t::try_t(const block_t& t)
: block_t(t),_is_finally(false)
{

}

try_t::~try_t()
{

}

void try_t::finalize()
{
    _is_finally = true;
    throw (*this);
}
