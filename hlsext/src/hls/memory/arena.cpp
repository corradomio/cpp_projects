//
// Created by Corrado Mio on 10/10/2015.
//

#include "../../../include/hls/memory/arena.hpp"
#include "../../../include/hls/memory/alloc.hpp"

using namespace hls::memory;

arena_t::arena_t(size_t page_size, bool expand)
: page_size(page_size), expand(expand), list(nullptr)
{ 

}


arena_t::~arena_t()
{
    free_all();
}


arena_t::page_t * arena_t::_alloc_page(size_t size)
{
    size_t npages = (size + page_size - 1)/page_size;
    page_t *page = (page_t*)hls::memory::alloc(sizeof(page_t) + npages*page_size);

    page->available = (hls::memory::alloc_size(page) - sizeof(page_t))/sizeof(size_t);
    page->next = list;

    list = page;

    return page;
}


void arena_t::free_all() {
    page_t *p, *n;
    for(p = list; p; p = n)
    {
        n = p->next;
        hls::memory::free(p);
    }
    list = nullptr;
}


void* arena_t::alloc(size_t count, size_t size) {
    return this->alloc(count*size);
}


void* arena_t::alloc(size_t size)
{
    page_t* p;
    size_t* ptr;
    size_t nslots = (size + sizeof(size_t) - 1)/sizeof(size_t) + 1;

    size = nslots*sizeof(size_t);

    // non puo' allocare piu' spazio di quanto disponibile in una pagina
    if (!expand && size > this->page_size)
        throw page_overflow();

    // cerca una pagina con abbastanza spazio
    for(p = list; p; p = p->next)
        if (p->available >= nslots)
            break;

    // non ha trovato una pagina con abbastanza spazio
    // ne alloca una nuova

    if (p == nullptr)
        p = _alloc_page(size);

    // alloca lo spazio nella pagina
    p->available -= nslots;
    ptr = &(p->page[p->available]);

    // salva l'informazione su quanto allocato
    ptr[0] = nslots;

    // ritorna il puntatore
    return (void*)(ptr+1);
}


void arena_t::free(void* p) {
    // non fa nulla
}


size_t arena_t::alloc_size(void* p) {
    size_t* ptr = ((size_t*)p) - 1;
    return ptr[0]*sizeof(size_t);
}