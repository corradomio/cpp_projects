//
// Created by Corrado Mio on 09/06/2024.
//

#ifndef STDX__MEMORY_H
#define STDX__MEMORY_H

#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace stdx {

    inline void * __cdecl memset(void *dst,int v, size_t bytes) {
        return ::memset(dst, v, bytes);
    }
    
    inline void * __cdecl memset8(void *dst,__int8 v, size_t bytes) {
        return ::memset(dst, v, bytes);
    }

    void * __cdecl memset16( void *dst, __int16 v, size_t count);
    void * __cdecl memset32( void *dst, __int32 v, size_t count);
    void * __cdecl memset64( void *dst, __int64 v, size_t count);
    void * __cdecl memset128(void *dst,__int128 v, size_t count);


    template<typename T>
    inline T* memfill(T* dst, T v, size_t count) {
        for(size_t i=0; i<count; ++i) dst[i] = v;
        return dst;
    }

}

#endif //STDX__MEMORY_H
