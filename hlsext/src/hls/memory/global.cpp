#include "../../../include/hls/memory/global.h"

using namespace hls::memory;


// --------------------------------------------------------------------------
// standard_t
// --------------------------------------------------------------------------

void* standard_t::alloc(size_t count, size_t size) {
    return ::calloc(count, size);
}

void* standard_t::alloc(size_t size) {
    return ::malloc(size);
}

void standard_t::free(void* p) {
    ::free(p);
}

size_t standard_t::alloc_size(void* p) {
    return hls::memory::msize(p);
}


// --------------------------------------------------------------------------
// default_t
// --------------------------------------------------------------------------

void* default_t::alloc(size_t count, size_t size) {
    return hls::memory::alloc(count, size);
}

void* default_t::alloc(size_t size) {
    return hls::memory::alloc(size);
}

void default_t::free(void* p) {
    hls::memory::free(p);
}

size_t default_t::alloc_size(void* p) {
    return hls::memory::alloc_size(p);
}


// --------------------------------------------------------------------------
// end
// --------------------------------------------------------------------------
