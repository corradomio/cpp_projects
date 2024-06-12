//
// Created by Corrado Mio on 08/06/2024.
//

#ifndef CHECK_CUDA_CTHREAD_H
#define CHECK_CUDA_CTHREAD_H

#include <intfloat.h>

namespace cudacpp {

    /// Used to specify grid/block dimensions
    struct dim_t {
        int x,y,z;

        dim_t(size_t x=1): dim_t(x,1,1) { };
        dim_t(size_t x, size_t y, size_t z=1): x(x),y(y),z(z) { }
        dim_t(const dim_t& dim): x(dim.x),y(dim.y),z(dim.z) { }

        dim_t& operator =(const dim_t& dim) {
            self.x = dim.x;
            self.y = dim.y;
            self.z = dim.z;
            return self;
        }
    };

    // struct cthread_t {
    //     const dim_t& gridDim;
    //     const dim_t& blockDim;
    //     const dim_t  blockIdx;
    //     const dim_t  threadIdx;
    //
    //     cthread_t(const dim_t& gridDim, const dim_t& blockDim, const dim_t blockIdx, const dim_t& threadIdx)
    //     : gridDim(gridDim), blockDim(blockDim), blockIdx(blockIdx), threadIdx(threadIdx)
    //     {
    //
    //     }
    // };

}

#endif //CHECK_CUDA_CTHREAD_H
