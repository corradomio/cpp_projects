//
// Created by Corrado Mio on 02/08/2023.
//

#ifndef CHECK_TRITON_EMU_TRITON_EMU_H
#define CHECK_TRITON_EMU_TRITON_EMU_H

#include <cstdint>
#include <cstdio>

namespace hls::triton {

    struct dim {
        std::size_t x, y, z;

        dim(): x(1), y(1), z(1) { }
        dim(size_t x_): x(x_), y(1), z(1) { }
        dim(size_t x_, size_t y_): x(x_), y(y_), z(1) { }
        dim(size_t x_, size_t y_, size_t z_): x(x_), y(y_), z(z_) { }

        dim(const dim& dim): x(dim.x), y(dim.y), z(dim.z) {}
    };

    struct thread_t {
        dim gridDim;
        dim blockIdx;
        dim blockDim;
        dim threadIdx;
        dim threadDim;

        thread_t(const dim& gridDim_, const dim& blockIdx_,
                 const dim& blockDim_, const dim& threadIdx_,
                 const dim& threadDim_)
        : gridDim(gridDim_), blockIdx(blockIdx_),
          blockDim(blockDim_), threadIdx(threadIdx_),
          threadDim(threadDim_)
        {

        }

        void run() {
            std::printf("[%zu,%zu,%zu] [%zu,%zu,%zu] Hello World\n",
                        blockIdx.z,
                        blockIdx.y,
                        blockIdx.x,
                        threadIdx.z,
                        threadIdx.y,
                        threadIdx.x
            );
        }
    };

    struct algorithm_t {
        dim gridDim;
        dim blockDim;
        dim threadDim;

        algorithm_t(const dim& blockDim_): gridDim(), blockDim(blockDim_) {

        }

        algorithm_t(const dim& gridDim, const dim& blockDim_): gridDim(gridDim), blockDim(blockDim_) {

        }

        algorithm_t(const dim& gridDim_, const dim& blockDim_, const dim& threadDim_)
        : gridDim(gridDim_), blockDim(blockDim_), threadDim(threadDim_) {

        }

        void run() {
            dim blockIdx;
            dim threadIdx;

            for(blockIdx.z=0; blockIdx.z < gridDim.z; blockIdx.z++)
                for(blockIdx.y=0; blockIdx.y < gridDim.y; blockIdx.y++)
                    for(blockIdx.x=0; blockIdx.x < gridDim.x; blockIdx.x++)
                        for(threadIdx.z=0; threadIdx.z < blockDim.z; threadIdx.z += threadDim.z)
                            for(threadIdx.y=0; threadIdx.y < blockDim.y; threadIdx.y += threadDim.y)
                                for(threadIdx.x=0; threadIdx.x < blockDim.x; threadIdx.x += threadDim.x) {
                                    thread_t t(gridDim, blockIdx, blockDim, threadIdx, threadDim);
                                    t.run();
                                }
        }

    };
};

#endif //CHECK_TRITON_EMU_TRITON_EMU_H
