//
// Created by Corrado Mio on 02/06/2024.
//

#ifndef CUDA_CUDACPP_H
#define CUDA_CUDACPP_H

#include <map>
#include <stdexcept>
#include <cuda.h>
#include <language.h>
#include <intfloat.h>


namespace cudacpp {

    // ----------------------------------------------------------------------
    // Errors
    // ----------------------------------------------------------------------

    class cuda_error: std::runtime_error {
        CUresult _error;
        mutable std::string _what;
    public:
        cuda_error(CUresult error);

        virtual const char* what() const noexcept override;
    };

    void check(CUresult res);
    // void check(cudaError_t res);

    // ----------------------------------------------------------------------
    // Utilities
    // ----------------------------------------------------------------------

    struct dim_t {
        int x,y,z;

        dim_t(uint16_t x=1): dim_t(x,1,1) { };
        dim_t(uint16_t x, uint16_t y, uint16_t z=1): x(x),y(y),z(z) { }
        dim_t(const dim_t& dim) = default;
        dim_t& operator =(const dim_t& dim) = default;
    };

    /// Map CUDA attribute name -> attribute id
    extern std::map<std::string, int> ATTRIBUTES;

    /// List of useful attributes, collected in a single step
    /// (all values can be retrieved using 'attribute(...)')
    struct cuda_attributes_t {
        bool initialized;

        struct {
            int major;
            int minor;
        } compute_capability;

        int total_memory_mb;
        int multiprocessors;
        int warp_size;
        int concurrent_kernels;

        struct {
            int threads_per_block;
            int blocks_per_multiprocessor;
            int threads_per_multiprocessor;
            int shared_memory_per_block;

            dim_t grid_dim;
            dim_t block_dim;
        } max;

    };

    // ----------------------------------------------------------------------
    // Main CUDA object
    // ----------------------------------------------------------------------
    // Used to initialize the CUDA support on a SINGLE device

    /// Main CUDA object, used to initialize the library and the default
    /// device. It is supported ONLY ONE device.
    struct cuda_t {
        int ordinal;
        CUdevice   dev;
        CUcontext  ctx;
        CUcontext pctx; // primary context

        mutable cuda_attributes_t attrs;

        cuda_t(const cuda_t& c) = delete;
        cuda_t& operator =(const cuda_t& c) = delete;
    public:
        cuda_t();
       ~cuda_t();

        /// Device name
        [[nodiscard]] std::string name() const;
        /// Collect in a single step several attributes
        [[nodiscard]] const cuda_attributes_t& attributes() const;

        /// Retrieve the attribute value.
        /// The list of attributes is available in ATTRIBUTES
        [[nodiscard]] int attribute(const std::string& name) const;
        [[nodiscard]] int attribute(int attrib) const;
    };

    /// Current device, if a 'cuda_t' object is created
    extern cuda_t* this_device;

    // ----------------------------------------------------------------------
    // Module object
    // ----------------------------------------------------------------------
    // Load a CUDA module and call a kernel
    //

    class module_t {
        CUmodule hmod;

        // disable copy constructor and assignment
        module_t(const module_t& m) = delete;
        module_t& operator =(const module_t& m) = delete;

        void unload();
    public:
        module_t(): hmod(nullptr){ }
        explicit module_t(const char* module_path);
        ~module_t();

    public:
        /// Launch a kernel.
        /// It is necessary to specify:
        ///
        ///     1) the grid size (n of blocks)
        ///     2) the block size (n of threads)
        ///     3) the kernel name
        ///     4) the list of parameters
        ///

        template<class... Types>
        void launch(const dim_t& block_dim, const char* name, Types... params) {
            size_t shared_mem = 0;
            self.launch({}, block_dim, shared_mem, name, params...);
        }

        template<class... Types>
        void launch(const dim_t& grid_dim, const dim_t& block_dim, const char* name, Types... params) {
            size_t shared_mem = 0;
            self.launch(grid_dim, block_dim, shared_mem, name, params...);
        }

        template<class... Types>
        void launch(const dim_t& grid_dim, const dim_t& block_dim, size_t shared_mem, const char* name, Types... params) {
            CUfunction hfun = nullptr;
            check(::cuModuleGetFunction(&hfun, self.hmod, name));
            void* args[] = {&params...};
            check(::cuLaunchKernel(
                hfun,
                grid_dim.x, grid_dim.y, grid_dim.z,
                block_dim.x, block_dim.y, block_dim.z,
                shared_mem, 0, args, nullptr
            ));
            check(::cuCtxSynchronize());
        }

    public:
        void load(const char* module_path);
    };

    // ----------------------------------------------------------------------
    // End
    // ----------------------------------------------------------------------

};


#endif //CUDA_CUDACPP_H
