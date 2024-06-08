//
// Created by Corrado Mio on 03/06/2024.
//

#include <memory>
#include <cuda.h>
#include "cudacpp/cudacpp.h"
#include "cudacpp/cudamem.h"
#include "language.h"

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
            case host:
                data = ::malloc(size);
                break;
            case device:
                check(::cuMemAlloc(&dptr, size));
                data = reinterpret_cast<void*>(dptr);
                break;
            case host_locked:
                check(::cuMemAllocHost(&data, size));
                break;
            case host_mapped:
                check(::cuMemHostAlloc(&data, size, CU_MEMHOSTALLOC_DEVICEMAP));
                break;
            case unified:
                check(::cuMemAllocManaged(&dptr, size, ::CU_MEM_ATTACH_GLOBAL));
                data = reinterpret_cast<void*>(dptr);
                break;
            default:
                throw std::runtime_error("Unsupported allocaton location");
                break;
        }

        return data;
    }

    void* cuda_free(void* data, loc_t loc) {
        CUdeviceptr dptr;
        if (data == nullptr)
            return nullptr;

        switch (loc) {
            case host:
                ::free(data);
                break;
            case device:
                dptr = reinterpret_cast<CUdeviceptr>(data);
                check(::cuMemFree(dptr));
                break;
            case host_locked:
                check(::cuMemFreeHost(data));
                break;
            case host_mapped:
                check(::cuMemFreeHost(data));
                break;
            case unified:
                dptr = reinterpret_cast<CUdeviceptr>(data);
                check(::cuMemFree(dptr));
            default:
                throw std::runtime_error("Unsupported allocation location");
                break;
        }
        return nullptr;
    }


    void* cuda_copy(void* dst, loc_t dst_loc, void* src, loc_t src_loc, size_t size) {
        // host, device, page_locked, unified
        bool src_device = (src_loc == device) || (src_loc == unified);
        bool dst_device = (dst_loc == device) || (dst_loc == unified);
        bool src_host   = (src_loc == host)   || (src_loc == host_mapped) || (src_loc == host_locked);
        bool dst_host   = (dst_loc == host)   || (dst_loc == host_mapped) || (dst_loc == host_locked);

        CUdeviceptr dptr;
        CUdeviceptr sptr;
        if (dst_loc == device_mapped && src_loc == host_mapped) {
           check(::cuMemHostGetDevicePointer(&dptr, src, 0));
           dst = reinterpret_cast<void*>(dptr);
        }
        elsif (dst_device && src_device) {
            dptr = reinterpret_cast<CUdeviceptr>(dst);
            sptr = reinterpret_cast<CUdeviceptr>(src);
            check(::cuMemcpyDtoD(dptr, sptr, size));
        }
        elsif (dst_device && src_host) {
            dptr = reinterpret_cast<CUdeviceptr>(dst);
            check(::cuMemcpyHtoD(dptr, src, size));
        }
        elsif (dst_host && src_device) {
            sptr = reinterpret_cast<CUdeviceptr>(src);
            check(::cuMemcpyDtoH(dst, sptr, size));
        }
        elsif (dst_host && src_host) {
            ::memcpy(dst, src, size);
        }
        else {
            throw cuda_error(CUDA_ERROR_ILLEGAL_ADDRESS);
        }
        return dst;
    }
}