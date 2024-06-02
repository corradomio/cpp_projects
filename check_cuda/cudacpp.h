//
// Created by Corrado Mio on 02/06/2024.
//

#ifndef CHECK_CUDA_CUDACPP_H
#define CHECK_CUDA_CUDACPP_H

#include <map>
#include <stdexcept>
#include "language.h"

namespace cudacpp {

    extern std::map<std::string, int> ATTRIBUTES;

    struct dim_t {
        int x,y,z;

        explicit dim_t(int x=1, int y=1, int z=1): x(x),y(y),z(z){ }
        explicit dim_t(const dim_t& dim): x(dim.x),y(dim.y),z(dim.z){ }

        dim_t& operator =(const dim_t& dim) {
            self.x = dim.x;
            self.y = dim.y;
            self.z = dim.z;
            return self;
        }
    };

    struct cuda_error: std::runtime_error {
        int result;
        cuda_error(int result);

        virtual const char* what() const noexcept override;
    };

    struct cuda_internal_t;

    // collection of useful attributes
    // (all values can be retrieved using 'attribute(...)')
    struct cuda_capabilities_t {
        int compute_capability_major;
        int compute_capability_minor;
        int total_memory_mb;
        int multiprocessors;
        int max_threads_per_block;
        int max_blocks_per_multiprocessor;
        int max_threads_per_multiprocessor;
        int max_shared_memory_per_block;
        int warp_size;
        int concurrent_kernels;

        dim_t max_grid_dim;
        dim_t max_block_dim;
    };

    class cuda_device_t {
        cuda_internal_t *_info;
    public:
        cuda_device_t();
       ~cuda_device_t();

        [[nodiscard]] std::string name() const;
        [[nodiscard]] cuda_capabilities_t capabilities() const;

        [[nodiscard]] int attribute(const std::string& name) const;
        [[nodiscard]] int attribute(int attrib) const;
    };
};


#endif //CHECK_CUDA_CUDACPP_H
