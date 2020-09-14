#include "../../../include/hls/memory/buffer.hpp"
#include "../../../include/hls/memory/alloc.hpp"


using namespace hls::memory;


static size_t ilog(size_t size) {
    size_t s = 1;
    size_t i = 0;
    while(s < size) {
        s += s;
        i += 1;
    }
    return i;
}



buffer_t::buffer_t() {
    for(int i=0; i<N; ++i)
        bucket[i] = nullptr;
    free_list = nullptr;
}


buffer_t::~buffer_t() {
    node_t *n, *t;

    for(int i=0; i<N; ++i) {
        for(n = bucket[i]; n; n = t) {
            t = n->n;
            hls::memory::free(n->p);
            delete n;
        }
    }

    for(n = free_list; n; n = t) {
        t = n->n;
        delete n;
    }
}


void* buffer_t::alloc(size_t count, size_t size) {
    return alloc(count*size);
}


void* buffer_t::alloc(size_t size) {
    void* p;
    node_t* n;
    size_t rest = size%sizeof(size_t);
    size = (size + (rest ? sizeof(size_t) - rest : 0));
    size_t ibucket = ilog(size);

    if (bucket[ibucket]) {
        n = bucket[ibucket];
        bucket[ibucket] = n->n;

        p = n->p;

        n->n = free_list;
        n->p = nullptr;
        free_list = n;
    }
    else {
        size = 1u << ibucket;
        p = hls::memory::alloc(size);
    }

    return p;
}


void buffer_t::free(void *p)
{
    size_t size = alloc_size(p);
    size_t ibucker = ilog(size);
    node_t* n;

    if(free_list) {
        n = free_list;
        free_list = n->n;
    } else {
        n = new node_t;
    }

    n->p = p;
    n->n = bucket[ibucker];
    bucket[ibucker] = n;
}


size_t buffer_t::alloc_size(void *p) {
    return hls::memory::alloc_size(p);
}

