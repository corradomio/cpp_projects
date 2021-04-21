//
// Created by Corrado Mio on 10/10/2015.
//

#ifndef HLS_MEMORY_GLOBAL_HPP
#define HLS_MEMORY_GLOBAL_HPP

#include <stddef.h>
#include <stdlib.h>
#include "stdx/exception.h"
#include "alloc.h"

namespace hls {
namespace memory {

    class standard_t {
    public:
        standard_t() { };
        ~standard_t() { };

        void* alloc(size_t count, size_t size);
        void* alloc(size_t size);
        void  free(void* p);

        size_t alloc_size(void* p);
    };


    class default_t {
    public:
        default_t() { };
        ~default_t() { };

        void* alloc(size_t count, size_t size);
        void* alloc(size_t size);
        void  free(void* p);

        size_t alloc_size(void* p);
    };

}}


#endif // HLS_MEMORY_GLOBAL_HPP
