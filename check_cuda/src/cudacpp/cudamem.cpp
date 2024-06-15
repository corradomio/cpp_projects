//
// Created by Corrado Mio on 03/06/2024.
//

#include <cstring>
#include <stdx/memory.h>
#include "cudacpp/cudacpp.h"
#include "cudacpp/cudamem.h"

#define NO_SIZE size_t(-1)


namespace cudacpp {

    // ----------------------------------------------------------------------
    // cuda_alloc
    // cuda_free
    // cuda_copy
    // ----------------------------------------------------------------------

    void* cuda_alloc(size_t n, size_t esize, loc_t loc) {
        CUdeviceptr dptr = 0;
        void* data = nullptr;
        size_t size = n*esize;

        switch(loc) {
            // host
            case host:
                data = ::malloc(size);
                break;
            case host_locked:
                check(::cuMemAllocHost(&data, size));
                break;
            case host_mapped:
                check(::cuMemHostAlloc(&data, size, CU_MEMHOSTALLOC_DEVICEMAP));
                break;

            // device
            case device:
                check(::cuMemAlloc(&dptr, size));
                data = reinterpret_cast<void*>(dptr);
                break;
            case unified:
                check(::cuMemAllocManaged(&dptr, size, ::CU_MEM_ATTACH_GLOBAL));
                data = reinterpret_cast<void*>(dptr);
                break;

            default:
                throw cuda_error(CUDA_ERROR_INVALID_VALUE);
        }

        return data;
    }

    void* cuda_free(void* data, loc_t loc) {
        CUdeviceptr dptr;

        if (data == nullptr)
            return nullptr;

        switch (loc) {
            // host
            case host:
                ::free(data);
                break;
            case host_locked:
                check(::cuMemFreeHost(data));
                break;
            case host_mapped:
                check(::cuMemFreeHost(data));
                break;

            // device
            case device:
                dptr = reinterpret_cast<CUdeviceptr>(data);
                check(::cuMemFree(dptr));
                break;
            case unified:
                dptr = reinterpret_cast<CUdeviceptr>(data);
                check(::cuMemFree(dptr));
                break;

            default:
                throw cuda_error(CUDA_ERROR_INVALID_VALUE);
        }
        return nullptr;
    }

    // host     host | host_locked
    //

    void* cuda_copy(void* dst, loc_t to_loc, void* src, loc_t src_loc, size_t size) {
        CUdeviceptr dptr;
        CUdeviceptr sptr;

        // host, device, page_locked, page_mapped, device_mapped, unified

        bool src_host   = (src_loc == host   || src_loc == host_locked   || src_loc == host_mapped);
        bool dst_host   = (to_loc == host || to_loc == host_locked || to_loc == host_mapped);
        bool src_device = (src_loc == device || src_loc == device_mapped);
        bool dst_device = (to_loc == device || to_loc == device_mapped);

        // special case
        if (dst_device && src_loc == host_mapped) {
          check(::cuMemHostGetDevicePointer(&dptr, src, 0));
          dst = reinterpret_cast<void*>(dptr);
        }
        elif (dst_host && src_host) {
            ::memcpy(dst, src, size);
        }
        elif (dst_device && src_device) {
            dptr = reinterpret_cast<CUdeviceptr>(dst);
            sptr = reinterpret_cast<CUdeviceptr>(src);
            check(::cuMemcpyDtoD(dptr, sptr, size));
        }
        elif (dst_host && src_device) {
            sptr = reinterpret_cast<CUdeviceptr>(src);
            check(::cuMemcpyDtoH(dst, sptr, size));
        }
        elif (dst_device && src_host) {
            dptr = reinterpret_cast<CUdeviceptr>(dst);
            check(::cuMemcpyHtoD(dptr, src, size));
        }
        elif (to_loc == unified || src_loc == unified) {
            dptr = reinterpret_cast<CUdeviceptr>(dst);
            sptr = reinterpret_cast<CUdeviceptr>(src);
            check(::cuMemcpy(dptr, sptr, size));
        }
        else {
            throw cuda_error(CUDA_ERROR_ILLEGAL_ADDRESS);
        }

        return dst;
    }

    // loc_t curr_loc = self._info->loc;
    // size_t bytes = self._info->n * sizeof(T);
    //
    // if (curr_loc == loc_t::unified) {
    //     // it is not necessary to move the memory
    // }
    // elif (curr_loc == to_loc) {
    //     // memory already located in the correct position
    // }
    // elif (curr_loc == loc_t::host_mapped && to_loc == loc_t::device) {
    //     self._info->ptr = self._data;
    //     void *copy = cuda_copy(nullptr, loc_t::device_mapped, self._data, curr_loc, bytes);
    //     self._data = reinterpret_cast<T*>(copy);
    //     self._info->loc = loc_t::device_mapped;
    // }
    // elif (curr_loc == loc_t::device_mapped && to_loc == loc_t::host) {
    //     self._data = reinterpret_cast<T*>(self._info->ptr);
    //     self._info->loc = loc_t::host_mapped;
    // }
    // elif (curr_loc != to_loc) {
    //     void *copy = cuda_alloc(self._info->n, sizeof(T), to_loc);
    //     cuda_copy(copy, to_loc, self._data, curr_loc, bytes);
    //     cuda_free(self._data, curr_loc);
    //     self._data = reinterpret_cast<T*>(copy);
    //     self._info->loc = to_loc;
    // }
    // else {
    //     // throw cuda_error(CUresult::CUDA_ERROR_ILLEGAL_ADDRESS);
    // }

    void* cuda_copy(loc_t to_loc, info_t* info, size_t esize) {
        loc_t curr_loc = info->loc;
        size_t bytes = info->n * esize;
        void* data = info->data;

        if (curr_loc == loc_t::unified) {
            // it is not necessary to move the memory
        }
        elif (curr_loc == to_loc) {
            // memory already located in the correct position
        }
        elif (curr_loc == loc_t::host_mapped && to_loc == loc_t::device) {
            info->ptr = data;
            info->data = cuda_copy(nullptr, loc_t::device_mapped, data, curr_loc, bytes);
            info->loc = loc_t::device_mapped;
        }
        elif (curr_loc == loc_t::device_mapped && to_loc == loc_t::host) {
            info->data = info->ptr;
            info->loc = loc_t::host_mapped;
        }
        elif (curr_loc != to_loc) {
            void *copy = cuda_alloc(info->n, esize, to_loc);
            cuda_copy(copy, to_loc, data, curr_loc, bytes);
            cuda_free(data, curr_loc);
            info->data = copy;
            info->loc = to_loc;
        }
        else {
            // throw cuda_error(CUresult::CUDA_ERROR_ILLEGAL_ADDRESS);
        }
        return info->data;
    }

    void cuda_fill(void* dst, loc_t loc, size_t size, int src, size_t src_size) {
        CUdeviceptr dptr;

        switch (loc) {
            case host:
            case host_locked:
            case host_mapped:
            case unified:
                switch(src_size) {
                    case 1: stdx::memset8( dst, src, size); break;
                    case 2: stdx::memset16(dst, src, size/2); break;
                    case 4: stdx::memset32(dst, src, size/4); break;
                    default:
                        throw cuda_error(CUDA_ERROR_INVALID_VALUE);
                }
                break;

            case device:
            case device_mapped:
                dptr = reinterpret_cast<CUdeviceptr>(dst);
                switch(src_size) {
                    case 1: check(::cuMemsetD8( dptr, src, size)); break;
                    case 2: check(::cuMemsetD16(dptr, src, size/2)); break;
                    case 4: check(::cuMemsetD32(dptr, src, size/4)); break;
                    default:
                        throw cuda_error(CUDA_ERROR_INVALID_VALUE);
                }
                break;
            default:
                throw cuda_error(CUDA_ERROR_INVALID_VALUE);
        }
    }

}