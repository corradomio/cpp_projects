//
// Created on 13/05/2015.
//

#ifndef HLS_MEMORY_ALLOC_HPP
#define HLS_MEMORY_ALLOC_HPP

#include "stdx/exception.h"

namespace hls {
namespace memory {

    const size_t MAGIC_NUM = 0x3A41C52AA25C14A3L;

    class out_of_memory : public stdx::exception {
    public:
        out_of_memory() { }
        out_of_memory(const out_of_memory& e) { }
        out_of_memory& operator =(const out_of_memory& e) { return *this; }
    };

    class memory_corrupted : public stdx::exception {
    public:
        memory_corrupted() { }
        memory_corrupted(const memory_corrupted& e) { }
        memory_corrupted& operator =(const memory_corrupted& e) { return *this; }
    };

    class invalid_ptr : public stdx::exception {
    public:
        invalid_ptr() { }
        invalid_ptr(const invalid_ptr& e) { }
        invalid_ptr& operator =(const invalid_ptr& e) { return *this; }
    };

    class already_free : public stdx::exception {
    public:
        already_free() { }
        already_free(const invalid_ptr& e) { }
        already_free& operator =(const already_free& e) { return *this; }
    };

    /**
     * Alternative to malloc/calloc/free with memory check
     */
    void* alloc(size_t size);
    void* alloc(size_t count, size_t size);
    void  free(void* p);

    size_t alloc_size(void* p);
    void*  validate_p(void* p);

    inline void* alloc(size_t count, size_t size) { return alloc(count*size); }

    size_t msize(void* p);

}};

#endif // HLS_MEMORY_ALLOC_HPP
