//
// Created  on 13/05/2015.
//

#include <stdlib.h>
#include "../../../include/hls/memory/alloc.hpp"

using namespace hls::memory;

/*
 *      +--------+
 *      | nslots | <-- real pointer
 *      +--------+
 *      |        | <-- used pointer
 *      |        |
 *      |        |
 *      :        :
 *      |        |
 *      |        |
 *      |        |
 *      +--------+
 *      | magic  | nslots ^ MAGIC_NUM
 *      +--------+
 *
 *      Maximum memory allocable: 2GB
 *      Size of a slot: 8 bytes
 */

static const size_t GB = 1024*1024*1024L;
static const size_t MAX_SLOTS = (2*GB)/sizeof(size_t);


/**
 * ATTENZIONE: e' STRETTAMENTE dipendente dall'implementazione usata
 * Al momento funziona SOLO per GNU GCC dopo una serie di analisi sull'allocazione
 */
size_t hls::memory::msize(void* p)
{
    size_t* ptr = ((size_t*)p)-1;
    return ptr[0]-11;
}


void* hls::memory::alloc(size_t size)
{
    // convert size in a number of slots of  sizeof(size_t) bytes
    size_t nslots = (size + sizeof(size_t) - 1)/sizeof(size_t);

    // add two slots: header and footer
    nslots += 2;

    // allocate the space (and fill it with zero))
    size_t* ptr = (size_t*)::calloc(nslots, sizeof(size_t));

    // check if the pointer is not null
    if (!ptr)
        throw hls::memory::out_of_memory();

    // initialize the header and the footer
    ptr[0] = nslots;
    ptr[nslots-1] = MAGIC_NUM ^ nslots;

    // return a poiter to the slot 1. The slot 0 is the header
    return (void*)(ptr+1);
}


static size_t* _validate(void* p)
{
    size_t *ptr, nslots;

    if (!p) return nullptr;

    // locate the pointer to the slot 0
    ptr = ((size_t*)p)-1;

    // the slot 0 contains the number of slots allocated
    nslots = ptr[0];

    // if nslots is a strage value, the pointer is not at the start of the block
    if (nslots < 0 || nslots > MAX_SLOTS)
        throw hls::memory::invalid_ptr();

    // if ptr[nslots-1] == ~nslots, the block is already freed
    if (~nslots == ptr[nslots-1])
        throw hls::memory::already_free();

    // possible write before or after the block
    if (ptr[nslots-1] != (MAGIC_NUM ^nslots))
        throw hls::memory::memory_corrupted();

    return ptr;
}


void hls::memory::free(void* p)
{
    size_t *ptr = _validate(p);
    size_t size = ptr[0];
    ptr[size-1] = ~size;
    ::free(ptr);
}


size_t hls::memory::alloc_size(void* p)
{
    size_t *ptr = _validate(p);
    return (ptr[0]-2)*sizeof(size_t);
}


void* hls::memory::validate_p(void* p)
{
    _validate(p);
    return p;
}
