//
// Created by Corrado Mio on 09/06/2024.
//
#include <cstring>
#include "stdx/memory.h"

namespace stdx {

    void * __cdecl memset16(void *dst, __int16 v, size_t count) {
        auto* p = (__int16*)dst;
        for (size_t i=0; i<count; ++i) p[i] = v;
        return dst;
    }

    void * __cdecl memset32(void *dst, __int32 v, size_t count) {
        auto* p = (__int32*)dst;
        for (size_t i=0; i<count; ++i) p[i] = v;
        return dst;
    }

    void * __cdecl memset64(void *dst, __int64 v, size_t count) {
        auto* p = (__int64*)dst;
        for (size_t i=0; i<count; ++i) p[i] = v;
        return dst;
    }

    void * __cdecl memset128(void *dst, __int128 v, size_t count) {
        auto* p = (__int128*)dst;
        for (size_t i=0; i<count; ++i) p[i] = v;
        return dst;
    }
}