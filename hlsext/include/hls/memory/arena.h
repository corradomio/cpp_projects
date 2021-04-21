//
// Created by Corrado Mio on 10/10/2015.
//

#ifndef HLS_MEMORY_ARENA_HPP
#define HLS_MEMORY_ARENA_HPP

#include <stddef.h>
#include "stdx/exception.h"

namespace hls {
namespace memory {

    class page_overflow : public stdx::exception_t {
    public:
        page_overflow() { }
        page_overflow(const page_overflow& e) { }
        page_overflow& operator =(const page_overflow& e) { return *this; }
    };


    class arena_t {

        struct page_t {
            page_t *next;       // prossima pagina
            size_t available;   // slot disponibili (uno slot e' sizeof(size_t) bytes
            size_t page[0];     // spazio allocato per la pagina
        };

        page_t* list;           // lista delle pagine allocate
        size_t  page_size;      // dimensione di una pagina (in byte)
        bool    expand;         // se puo' allocare multipli della dimensione di una pagina

        page_t * _alloc_page(size_t size);
    public:
        arena_t(size_t page_size, bool expand=false);
        ~arena_t();

        void free_all();

        void* alloc(size_t count, size_t size);
        void* alloc(size_t size);
        void  free(void* p);

        size_t alloc_size(void* p);
    };

}}


#endif // HLS_MEMORY_ARENA_HPP
