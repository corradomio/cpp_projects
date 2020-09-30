//
// Created by Corrado Mio on 29/09/2020.
//

#include <memory>
#include <gc/gc_cpp.h>
#include <vector>

#ifndef BOHEM_ALLOCATOR_H
#define BOHEM_ALLOCATOR_H

namespace bohem {

    template<typename T>
    struct allocator {
        typedef T value_type;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        allocator() { }
        ~allocator() { }

        T* allocate(std::size_t n) { return new (GC) T[n]; }
        void deallocate( T* p, std::size_t n ) { }
    };

}

#endif //CHECK_GC_GC_ALLOCATOR_H
