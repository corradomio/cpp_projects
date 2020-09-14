
#ifndef HLS_MEMORY_BUFFER_HPP
#define HLS_MEMORY_BUFFER_HPP

#include <stddef.h>

namespace hls {
namespace memory {

    class buffer_t
    {
        const size_t N = 32;

        struct node_t {
            node_t *n;
            void*p;
        };

        node_t* bucket[32];
        node_t* free_list;

    public:
        buffer_t();
        ~buffer_t();

        void* alloc(size_t count, size_t size);
        void* alloc(size_t size);
        void  free(void* p);

        size_t alloc_size(void* p);
    };

}};

#endif // HLS_MEMORY_BUFFER_HPP